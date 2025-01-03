//===- SLPVectorizer.cpp - A bottom up SLP Vectorizer ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SLPVectorizer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CmpPredicate.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/VFABIDemangler.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstddef>
#include <iterator>
#include <set>
#include <tuple>

using namespace llvm;
using namespace slpvectorizer;

#define DEBUG_TYPE "SLP"

static cl::opt<bool>
    RunSLPVectorization("vectorize-slp", cl::init(true), cl::Hidden,
                        cl::desc("Run the SLP vectorization passes"));

static cl::opt<bool>
    SLPReVec("slp-revec", cl::init(false), cl::Hidden,
             cl::desc("Enable vectorization for wider vector utilization"));

static cl::opt<int>
MaxVectorRegSizeOption("slp-max-reg-size", cl::init(128), cl::Hidden,
    cl::desc("Attempt to vectorize for this register size in bits"));

static cl::opt<unsigned>
MaxVFOption("slp-max-vf", cl::init(0), cl::Hidden,
    cl::desc("Maximum SLP vectorization factor (0=unlimited)"));

static cl::opt<int> MinVectorRegSizeOption(
    "slp-min-reg-size", cl::init(128), cl::Hidden,
    cl::desc("Attempt to vectorize for this register size in bits"));

static cl::opt<unsigned> RecursionMaxDepth(
    "slp-recursion-max-depth", cl::init(12), cl::Hidden,
    cl::desc("Limit the recursion depth when building a vectorizable tree"));

static cl::opt<bool> VectorizeNonPowerOf2(
    "slp-vectorize-non-power-of-2", cl::init(false), cl::Hidden,
    cl::desc("Try to vectorize with non-power-of-2 number of elements."));

/// Predicate for the element types that the SLP vectorizer supports.
///
/// The most important thing to filter here are types which are invalid in LLVM
/// vectors. We also filter target specific types which have absolutely no
/// meaningful vectorization path such as x86_fp80 and ppc_f128. This just
/// avoids spending time checking the cost model and realizing that they will
/// be inevitably scalarized.
static bool isValidElementType(Type *Ty) {
  // TODO: Support ScalableVectorType.
  if (SLPReVec && isa<FixedVectorType>(Ty))
    Ty = Ty->getScalarType();
  return VectorType::isValidElementType(Ty) && !Ty->isX86_FP80Ty() &&
         !Ty->isPPC_FP128Ty();
}

/// \returns True if the value is a constant (but not globals/constant
/// expressions).
static bool isConstant(Value *V) {
  return isa<Constant>(V) && !isa<ConstantExpr, GlobalValue>(V);
}

/// Checks if \p V is one of vector-like instructions, i.e. undef,
/// insertelement/extractelement with constant indices for fixed vector type or
/// extractvalue instruction.
static bool isVectorLikeInstWithConstOps(Value *V) {
  if (!isa<InsertElementInst, ExtractElementInst>(V) &&
      !isa<ExtractValueInst, UndefValue>(V))
    return false;
  auto *I = dyn_cast<Instruction>(V);
  if (!I || isa<ExtractValueInst>(I))
    return true;
  if (!isa<FixedVectorType>(I->getOperand(0)->getType()))
    return false;
  if (isa<ExtractElementInst>(I))
    return isConstant(I->getOperand(1));
  assert(isa<InsertElementInst>(V) && "Expected only insertelement.");
  return isConstant(I->getOperand(2));
}

namespace {
class InstructionsState {
  /// The main/alternate instruction. MainOp is also VL0.
  Instruction *MainOp = nullptr;
  Instruction *AltOp = nullptr;

public:
  /// The main/alternate opcodes for the list of instructions.
  unsigned getOpcode() const {
    return MainOp ? MainOp->getOpcode() : 0;
  }

  InstructionsState() = delete;
  InstructionsState(Instruction *MainOp, Instruction *AltOp)
      : MainOp(MainOp), AltOp(AltOp) {}
  static InstructionsState invalid() { return {nullptr, nullptr}; }
};

} // end anonymous namespace

/// \returns true if \p Opcode is allowed as part of the main/alternate
/// instruction for SLP vectorization.
///
/// Example of unsupported opcode is SDIV that can potentially cause UB if the
/// "shuffled out" lane would result in division by zero.
static bool isValidForAlternation(unsigned Opcode) {
  if (Instruction::isIntDivRem(Opcode))
    return false;

  return true;
}

static InstructionsState getSameOpcode(ArrayRef<Value *> VL,
                                       const TargetLibraryInfo &TLI);

/// Checks if the provided operands of 2 cmp instructions are compatible, i.e.
/// compatible instructions or constants, or just some other regular values.
static bool areCompatibleCmpOps(Value *BaseOp0, Value *BaseOp1, Value *Op0,
                                Value *Op1, const TargetLibraryInfo &TLI) {
  return (isConstant(BaseOp0) && isConstant(Op0)) ||
         (isConstant(BaseOp1) && isConstant(Op1)) ||
         (!isa<Instruction>(BaseOp0) && !isa<Instruction>(Op0) &&
          !isa<Instruction>(BaseOp1) && !isa<Instruction>(Op1)) ||
         BaseOp0 == Op0 || BaseOp1 == Op1 ||
         getSameOpcode({BaseOp0, Op0}, TLI).getOpcode() ||
         getSameOpcode({BaseOp1, Op1}, TLI).getOpcode();
}

/// \returns true if a compare instruction \p CI has similar "look" and
/// same predicate as \p BaseCI, "as is" or with its operands and predicate
/// swapped, false otherwise.
static bool isCmpSameOrSwapped(const CmpInst *BaseCI, const CmpInst *CI,
                               const TargetLibraryInfo &TLI) {
  assert(BaseCI->getOperand(0)->getType() == CI->getOperand(0)->getType() &&
         "Assessing comparisons of different types?");
  CmpInst::Predicate BasePred = BaseCI->getPredicate();
  CmpInst::Predicate Pred = CI->getPredicate();
  CmpInst::Predicate SwappedPred = CmpInst::getSwappedPredicate(Pred);

  Value *BaseOp0 = BaseCI->getOperand(0);
  Value *BaseOp1 = BaseCI->getOperand(1);
  Value *Op0 = CI->getOperand(0);
  Value *Op1 = CI->getOperand(1);

  return (BasePred == Pred &&
          areCompatibleCmpOps(BaseOp0, BaseOp1, Op0, Op1, TLI)) ||
         (BasePred == SwappedPred &&
          areCompatibleCmpOps(BaseOp0, BaseOp1, Op1, Op0, TLI));
}

/// \returns analysis of the Instructions in \p VL described in
/// InstructionsState, the Opcode that we suppose the whole list
/// could be vectorized even if its structure is diverse.
static InstructionsState getSameOpcode(ArrayRef<Value *> VL,
                                       const TargetLibraryInfo &TLI) {
  // Make sure these are all Instructions.
  if (!all_of(VL, IsaPred<Instruction, PoisonValue>))
    return InstructionsState::invalid();

  auto *It = find_if(VL, IsaPred<Instruction>);
  if (It == VL.end())
    return InstructionsState::invalid();

  Value *V = *It;
  unsigned InstCnt = std::count_if(It, VL.end(), IsaPred<Instruction>);
  if ((VL.size() > 2 && !isa<PHINode>(V) && InstCnt < VL.size() / 2) ||
      (VL.size() == 2 && InstCnt < 2))
    return InstructionsState::invalid();

  bool IsCastOp = isa<CastInst>(V);
  bool IsBinOp = isa<BinaryOperator>(V);
  bool IsCmpOp = isa<CmpInst>(V);
  CmpInst::Predicate BasePred =
      IsCmpOp ? cast<CmpInst>(V)->getPredicate() : CmpInst::BAD_ICMP_PREDICATE;
  unsigned Opcode = cast<Instruction>(V)->getOpcode();
  unsigned AltOpcode = Opcode;
  unsigned AltIndex = std::distance(VL.begin(), It);

  bool SwappedPredsCompatible = [&]() {
    if (!IsCmpOp)
      return false;
    SetVector<unsigned> UniquePreds, UniqueNonSwappedPreds;
    UniquePreds.insert(BasePred);
    UniqueNonSwappedPreds.insert(BasePred);
    for (Value *V : VL) {
      auto *I = dyn_cast<CmpInst>(V);
      if (!I)
        return false;
      CmpInst::Predicate CurrentPred = I->getPredicate();
      CmpInst::Predicate SwappedCurrentPred =
          CmpInst::getSwappedPredicate(CurrentPred);
      UniqueNonSwappedPreds.insert(CurrentPred);
      if (!UniquePreds.contains(CurrentPred) &&
          !UniquePreds.contains(SwappedCurrentPred))
        UniquePreds.insert(CurrentPred);
    }
    // Total number of predicates > 2, but if consider swapped predicates
    // compatible only 2, consider swappable predicates as compatible opcodes,
    // not alternate.
    return UniqueNonSwappedPreds.size() > 2 && UniquePreds.size() == 2;
  }();
  // Check for one alternate opcode from another BinaryOperator.
  // TODO - generalize to support all operators (types, calls etc.).
  auto *IBase = cast<Instruction>(V);
  Intrinsic::ID BaseID = 0;
  SmallVector<VFInfo> BaseMappings;
  if (auto *CallBase = dyn_cast<CallInst>(IBase)) {
    BaseID = getVectorIntrinsicIDForCall(CallBase, &TLI);
    BaseMappings = VFDatabase(*CallBase).getMappings(*CallBase);
    if (!isTriviallyVectorizable(BaseID) && BaseMappings.empty())
      return InstructionsState::invalid();
  }
  bool AnyPoison = InstCnt != VL.size();
  for (int Cnt = 0, E = VL.size(); Cnt < E; Cnt++) {
    auto *I = dyn_cast<Instruction>(VL[Cnt]);
    if (!I)
      continue;

    // Cannot combine poison and divisions.
    // TODO: do some smart analysis of the CallInsts to exclude divide-like
    // intrinsics/functions only.
    if (AnyPoison && (I->isIntDivRem() || I->isFPDivRem() || isa<CallInst>(I)))
      return InstructionsState::invalid();
    unsigned InstOpcode = I->getOpcode();
    if (IsBinOp && isa<BinaryOperator>(I)) {
      if (InstOpcode == Opcode || InstOpcode == AltOpcode)
        continue;
      if (Opcode == AltOpcode && isValidForAlternation(InstOpcode) &&
          isValidForAlternation(Opcode)) {
        AltOpcode = InstOpcode;
        AltIndex = Cnt;
        continue;
      }
    } else if (IsCastOp && isa<CastInst>(I)) {
      Value *Op0 = IBase->getOperand(0);
      Type *Ty0 = Op0->getType();
      Value *Op1 = I->getOperand(0);
      Type *Ty1 = Op1->getType();
      if (Ty0 == Ty1) {
        if (InstOpcode == Opcode || InstOpcode == AltOpcode)
          continue;
        if (Opcode == AltOpcode) {
          assert(isValidForAlternation(Opcode) &&
                 isValidForAlternation(InstOpcode) &&
                 "Cast isn't safe for alternation, logic needs to be updated!");
          AltOpcode = InstOpcode;
          AltIndex = Cnt;
          continue;
        }
      }
    } else if (auto *Inst = dyn_cast<CmpInst>(VL[Cnt]); Inst && IsCmpOp) {
      auto *BaseInst = cast<CmpInst>(V);
      Type *Ty0 = BaseInst->getOperand(0)->getType();
      Type *Ty1 = Inst->getOperand(0)->getType();
      if (Ty0 == Ty1) {
        assert(InstOpcode == Opcode && "Expected same CmpInst opcode.");
        assert(InstOpcode == AltOpcode &&
               "Alternate instructions are only supported by BinaryOperator "
               "and CastInst.");
        // Check for compatible operands. If the corresponding operands are not
        // compatible - need to perform alternate vectorization.
        CmpInst::Predicate CurrentPred = Inst->getPredicate();
        CmpInst::Predicate SwappedCurrentPred =
            CmpInst::getSwappedPredicate(CurrentPred);

        if ((E == 2 || SwappedPredsCompatible) &&
            (BasePred == CurrentPred || BasePred == SwappedCurrentPred))
          continue;

        if (isCmpSameOrSwapped(BaseInst, Inst, TLI))
          continue;
        auto *AltInst = cast<CmpInst>(VL[AltIndex]);
        if (AltIndex) {
          if (isCmpSameOrSwapped(AltInst, Inst, TLI))
            continue;
        } else if (BasePred != CurrentPred) {
          assert(
              isValidForAlternation(InstOpcode) &&
              "CmpInst isn't safe for alternation, logic needs to be updated!");
          AltIndex = Cnt;
          continue;
        }
        CmpInst::Predicate AltPred = AltInst->getPredicate();
        if (BasePred == CurrentPred || BasePred == SwappedCurrentPred ||
            AltPred == CurrentPred || AltPred == SwappedCurrentPred)
          continue;
      }
    } else if (InstOpcode == Opcode) {
      assert(InstOpcode == AltOpcode &&
             "Alternate instructions are only supported by BinaryOperator and "
             "CastInst.");
      if (auto *Gep = dyn_cast<GetElementPtrInst>(I)) {
        if (Gep->getNumOperands() != 2 ||
            Gep->getOperand(0)->getType() != IBase->getOperand(0)->getType())
          return InstructionsState::invalid();
      } else if (auto *EI = dyn_cast<ExtractElementInst>(I)) {
        if (!isVectorLikeInstWithConstOps(EI))
          return InstructionsState::invalid();
      } else if (auto *LI = dyn_cast<LoadInst>(I)) {
        auto *BaseLI = cast<LoadInst>(IBase);
        if (!LI->isSimple() || !BaseLI->isSimple())
          return InstructionsState::invalid();
      } else if (auto *Call = dyn_cast<CallInst>(I)) {
        auto *CallBase = cast<CallInst>(IBase);
        if (Call->getCalledFunction() != CallBase->getCalledFunction())
          return InstructionsState::invalid();
        if (Call->hasOperandBundles() &&
            (!CallBase->hasOperandBundles() ||
             !std::equal(Call->op_begin() + Call->getBundleOperandsStartIndex(),
                         Call->op_begin() + Call->getBundleOperandsEndIndex(),
                         CallBase->op_begin() +
                             CallBase->getBundleOperandsStartIndex())))
          return InstructionsState::invalid();
        Intrinsic::ID ID = getVectorIntrinsicIDForCall(Call, &TLI);
        if (ID != BaseID)
          return InstructionsState::invalid();
        if (!ID) {
          SmallVector<VFInfo> Mappings = VFDatabase(*Call).getMappings(*Call);
          if (Mappings.size() != BaseMappings.size() ||
              Mappings.front().ISA != BaseMappings.front().ISA ||
              Mappings.front().ScalarName != BaseMappings.front().ScalarName ||
              Mappings.front().VectorName != BaseMappings.front().VectorName ||
              Mappings.front().Shape.VF != BaseMappings.front().Shape.VF ||
              Mappings.front().Shape.Parameters !=
                  BaseMappings.front().Shape.Parameters)
            return InstructionsState::invalid();
        }
      }
      continue;
    }
    return InstructionsState::invalid();
  }

  return InstructionsState(cast<Instruction>(V),
                           cast<Instruction>(VL[AltIndex]));
}

namespace llvm {

namespace slpvectorizer {

/// Bottom Up SLP Vectorizer.
class BoUpSLP {
public:
  using ValueList = SmallVector<Value *, 8>;
  using ValueSet = SmallPtrSet<Value *, 16>;

  BoUpSLP(Function *Func, ScalarEvolution *Se, TargetTransformInfo *Tti,
          TargetLibraryInfo *TLi, AAResults *Aa, LoopInfo *Li,
          DominatorTree *Dt, AssumptionCache *AC, DemandedBits *DB,
          const DataLayout *DL, OptimizationRemarkEmitter *ORE)
      : BatchAA(*Aa), F(Func), SE(Se), TTI(Tti), TLI(TLi), LI(Li), DT(Dt),
        AC(AC), DB(DB), DL(DL), ORE(ORE),
        Builder(Se->getContext(), TargetFolder(*DL)) {
    CodeMetrics::collectEphemeralValues(F, AC, EphValues);
    // Use the vector register size specified by the target unless overridden
    // by a command-line option.
    // TODO: It would be better to limit the vectorization factor based on
    //       data type rather than just register size. For example, x86 AVX has
    //       256-bit registers, but it does not support integer operations
    //       at that width (that requires AVX2).
    if (MaxVectorRegSizeOption.getNumOccurrences())
      MaxVecRegSize = MaxVectorRegSizeOption;
    else
      MaxVecRegSize =
          TTI->getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector)
              .getFixedValue();

    if (MinVectorRegSizeOption.getNumOccurrences())
      MinVecRegSize = MinVectorRegSizeOption;
    else
      MinVecRegSize = TTI->getMinVectorRegisterBitWidth();
  }

  /// \return The vector element size in bits to use when vectorizing the
  /// expression tree ending at \p V. If V is a store, the size is the width of
  /// the stored value. Otherwise, the size is the width of the largest loaded
  /// value reaching V. This method is used by the vectorizer to calculate
  /// vectorization factors.
  unsigned getVectorElementSize(Value *V);

  // \returns maximum vector register size as set by TTI or overridden by cl::opt.
  unsigned getMaxVecRegSize() const {
    return MaxVecRegSize;
  }

  // \returns minimum vector register size as set by cl::opt.
  unsigned getMinVecRegSize() const {
    return MinVecRegSize;
  }

  unsigned getMinVF(unsigned Sz) const {
    return std::max(2U, getMinVecRegSize() / Sz);
  }

  unsigned getMaximumVF(unsigned ElemWidth, unsigned Opcode) const {
    unsigned MaxVF = MaxVFOption.getNumOccurrences() ?
      MaxVFOption : TTI->getMaximumVF(ElemWidth, Opcode);
    return MaxVF ? MaxVF : UINT_MAX;
  }

  /// Checks if the instruction is marked for deletion.
  bool isDeleted(Instruction *I) const { return DeletedInstructions.count(I); }

  /// Clear the list of the analyzed reduction root instructions.
  void clearReductionData() {
    AnalyzedReductionsRoots.clear();
    AnalyzedReductionVals.clear();
    AnalyzedMinBWVals.clear();
  }

  /// Maps a value to the proposed vectorizable size.
  SmallDenseMap<Value *, unsigned> InstrElementSize;

  // Cache for pointerMayBeCaptured calls inside AA.  This is preserved
  // globally through SLP because we don't perform any action which
  // invalidates capture results.
  BatchAAResults BatchAA;

  /// Temporary store for deleted instructions. Instructions will be deleted
  /// eventually when the BoUpSLP is destructed.  The deferral is required to
  /// ensure that there are no incorrect collisions in the AliasCache, which
  /// can happen if a new instruction is allocated at the same address as a
  /// previously deleted instruction.
  DenseSet<Instruction *> DeletedInstructions;

  /// Set of the instruction, being analyzed already for reductions.
  SmallPtrSet<Instruction *, 16> AnalyzedReductionsRoots;

  /// Set of hashes for the list of reduction values already being analyzed.
  DenseSet<size_t> AnalyzedReductionVals;

  /// Values, already been analyzed for mininmal bitwidth and found to be
  /// non-profitable.
  DenseSet<Value *> AnalyzedMinBWVals;

  /// Values used only by @llvm.assume calls.
  SmallPtrSet<const Value *, 32> EphValues;

  // Analysis and block reference.
  Function *F;
  ScalarEvolution *SE;
  TargetTransformInfo *TTI;
  TargetLibraryInfo *TLI;
  LoopInfo *LI;
  DominatorTree *DT;
  AssumptionCache *AC;
  DemandedBits *DB;
  const DataLayout *DL;
  OptimizationRemarkEmitter *ORE;

  unsigned MaxVecRegSize; // This is set by TTI or overridden by cl::opt.
  unsigned MinVecRegSize; // Set by cl::opt (default: 128).

  /// Instruction builder to construct the vectorized tree.
  IRBuilder<TargetFolder> Builder;
};

} // end namespace slpvectorizer

} // end namespace llvm

unsigned BoUpSLP::getVectorElementSize(Value *V) {
  // If V is a store, just return the width of the stored value (or value
  // truncated just before storing) without traversing the expression tree.
  // This is the common case.
  if (auto *Store = dyn_cast<StoreInst>(V))
    return DL->getTypeSizeInBits(Store->getValueOperand()->getType());

  if (auto *IEI = dyn_cast<InsertElementInst>(V))
    return getVectorElementSize(IEI->getOperand(1));

  auto E = InstrElementSize.find(V);
  if (E != InstrElementSize.end())
    return E->second;

  // If V is not a store, we can traverse the expression tree to find loads
  // that feed it. The type of the loaded value may indicate a more suitable
  // width than V's type. We want to base the vector element size on the width
  // of memory operations where possible.
  SmallVector<std::tuple<Instruction *, BasicBlock *, unsigned>> Worklist;
  SmallPtrSet<Instruction *, 16> Visited;
  if (auto *I = dyn_cast<Instruction>(V)) {
    Worklist.emplace_back(I, I->getParent(), 0);
    Visited.insert(I);
  }

  // Traverse the expression tree in bottom-up order looking for loads. If we
  // encounter an instruction we don't yet handle, we give up.
  auto Width = 0u;
  Value *FirstNonBool = nullptr;
  while (!Worklist.empty()) {
    auto [I, Parent, Level] = Worklist.pop_back_val();

    // We should only be looking at scalar instructions here. If the current
    // instruction has a vector type, skip.
    auto *Ty = I->getType();
    if (isa<VectorType>(Ty))
      continue;
    if (Ty != Builder.getInt1Ty() && !FirstNonBool)
      FirstNonBool = I;
    if (Level > RecursionMaxDepth)
      continue;

    // If the current instruction is a load, update MaxWidth to reflect the
    // width of the loaded value.
    if (isa<LoadInst, ExtractElementInst, ExtractValueInst>(I))
      Width = std::max<unsigned>(Width, DL->getTypeSizeInBits(Ty));

    // Otherwise, we need to visit the operands of the instruction. We only
    // handle the interesting cases from buildTree here. If an operand is an
    // instruction we haven't yet visited and from the same basic block as the
    // user or the use is a PHI node, we add it to the worklist.
    else if (isa<PHINode, CastInst, GetElementPtrInst, CmpInst, SelectInst,
                 BinaryOperator, UnaryOperator>(I)) {
      for (Use &U : I->operands()) {
        if (auto *J = dyn_cast<Instruction>(U.get()))
          if (Visited.insert(J).second &&
              (isa<PHINode>(I) || J->getParent() == Parent)) {
            Worklist.emplace_back(J, J->getParent(), Level + 1);
            continue;
          }
        if (!FirstNonBool && U.get()->getType() != Builder.getInt1Ty())
          FirstNonBool = U.get();
      }
    } else {
      break;
    }
  }

  // If we didn't encounter a memory access in the expression tree, or if we
  // gave up for some reason, just return the width of V. Otherwise, return the
  // maximum width we found.
  if (!Width) {
    if (V->getType() == Builder.getInt1Ty() && FirstNonBool)
      V = FirstNonBool;
    Width = DL->getTypeSizeInBits(V->getType());
  }

  for (Instruction *I : Visited)
    InstrElementSize[I] = Width;

  return Width;
}

PreservedAnalyses SLPVectorizerPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto *SE = &AM.getResult<ScalarEvolutionAnalysis>(F);
  auto *TTI = &AM.getResult<TargetIRAnalysis>(F);
  auto *TLI = AM.getCachedResult<TargetLibraryAnalysis>(F);
  auto *AA = &AM.getResult<AAManager>(F);
  auto *LI = &AM.getResult<LoopAnalysis>(F);
  auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
  auto *AC = &AM.getResult<AssumptionAnalysis>(F);
  auto *DB = &AM.getResult<DemandedBitsAnalysis>(F);
  auto *ORE = &AM.getResult<OptimizationRemarkEmitterAnalysis>(F);

  bool Changed = runImpl(F, SE, TTI, TLI, AA, LI, DT, AC, DB, ORE);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

bool SLPVectorizerPass::runImpl(Function &F, ScalarEvolution *SE_,
                                TargetTransformInfo *TTI_,
                                TargetLibraryInfo *TLI_, AAResults *AA_,
                                LoopInfo *LI_, DominatorTree *DT_,
                                AssumptionCache *AC_, DemandedBits *DB_,
                                OptimizationRemarkEmitter *ORE_) {
  if (!RunSLPVectorization)
    return false;
  SE = SE_;
  TTI = TTI_;
  TLI = TLI_;
  AA = AA_;
  LI = LI_;
  DT = DT_;
  AC = AC_;
  DB = DB_;
  DL = &F.getDataLayout();

  Stores.clear();
  GEPs.clear();
  bool Changed = false;

  // If the target claims to have no vector registers don't attempt
  // vectorization.
  if (!TTI->getNumberOfRegisters(TTI->getRegisterClassForType(true))) {
    LLVM_DEBUG(
        dbgs() << "SLP: Didn't find any vector registers for target, abort.\n");
    return false;
  }

  // Don't vectorize when the attribute NoImplicitFloat is used.
  if (F.hasFnAttribute(Attribute::NoImplicitFloat))
    return false;

  LLVM_DEBUG(dbgs() << "SLP: Analyzing blocks in " << F.getName() << ".\n");

  // Use the bottom up slp vectorizer to construct chains that start with
  // store instructions.
  BoUpSLP R(&F, SE, TTI, TLI, AA, LI, DT, AC, DB, DL, ORE_);

  // A general note: the vectorizer must use BoUpSLP::eraseInstruction() to
  // delete instructions.

  // Update DFS numbers now so that we can use them for ordering.
  DT->updateDFSNumbers();

  // Scan the blocks in the function in post order.
  for (auto *BB : post_order(&F.getEntryBlock())) {
    if (BB->isEHPad() || isa_and_nonnull<UnreachableInst>(BB->getTerminator()))
      continue;

    // Start new block - clear the list of reduction roots.
    R.clearReductionData();
    collectSeedInstructions(BB);

    // Vectorize trees that end at stores.
    if (!Stores.empty()) {
      LLVM_DEBUG(dbgs() << "SLP: Found stores for " << Stores.size()
                        << " underlying objects.\n");
      Changed |= vectorizeStoreChains(R);
    }
  }

  return Changed;
}

bool SLPVectorizerPass::vectorizeStores(
    ArrayRef<StoreInst *> Stores, BoUpSLP &R,
    DenseSet<std::tuple<Value *, Value *, Value *, Value *, unsigned>>
        &Visited) {
  // We may run into multiple chains that merge into a single chain. We mark the
  // stores that we vectorized so that we don't visit the same store twice.
  BoUpSLP::ValueSet VectorizedStores;
  bool Changed = false;

  struct StoreDistCompare {
    bool operator()(const std::pair<unsigned, int> &Op1,
                    const std::pair<unsigned, int> &Op2) const {
      return Op1.second < Op2.second;
    }
  };
  // A set of pairs (index of store in Stores array ref, Distance of the store
  // address relative to base store address in units).
  using StoreIndexToDistSet =
      std::set<std::pair<unsigned, int>, StoreDistCompare>;
  auto TryToVectorize = [&](const StoreIndexToDistSet &Set) {
    int PrevDist = -1;
    BoUpSLP::ValueList Operands;
    // Collect the chain into a list.
    for (auto [Idx, Data] : enumerate(Set)) {
      if (Operands.empty() || Data.second - PrevDist == 1) {
        Operands.push_back(Stores[Data.first]);
        PrevDist = Data.second;
        if (Idx != Set.size() - 1)
          continue;
      }
      auto E = make_scope_exit([&, &DataVar = Data]() {
        Operands.clear();
        Operands.push_back(Stores[DataVar.first]);
        PrevDist = DataVar.second;
      });

      if (Operands.size() <= 1 ||
          !Visited
               .insert({Operands.front(),
                        cast<StoreInst>(Operands.front())->getValueOperand(),
                        Operands.back(),
                        cast<StoreInst>(Operands.back())->getValueOperand(),
                        Operands.size()})
               .second)
        continue;

      unsigned MaxVecRegSize = R.getMaxVecRegSize();
      unsigned EltSize = R.getVectorElementSize(Operands[0]);
      unsigned MaxElts = llvm::bit_floor(MaxVecRegSize / EltSize);

      unsigned MaxVF =
          std::min(R.getMaximumVF(EltSize, Instruction::Store), MaxElts);
      auto *Store = cast<StoreInst>(Operands[0]);
      Type *StoreTy = Store->getValueOperand()->getType();
      Type *ValueTy = StoreTy;
      if (auto *Trunc = dyn_cast<TruncInst>(Store->getValueOperand()))
        ValueTy = Trunc->getSrcTy();
      unsigned MinVF = std::max<unsigned>(
          2, PowerOf2Ceil(TTI->getStoreMinimumVF(
                 R.getMinVF(DL->getTypeStoreSizeInBits(StoreTy)), StoreTy,
                 ValueTy)));

      if (MaxVF < MinVF) {
        LLVM_DEBUG(dbgs() << "SLP: Vectorization infeasible as MaxVF (" << MaxVF
                          << ") < "
                          << "MinVF (" << MinVF << ")\n");
        continue;
      }

      unsigned NonPowerOf2VF = 0;
      if (VectorizeNonPowerOf2) {
      }
    }
  };

  return Changed;
}

void SLPVectorizerPass::collectSeedInstructions(BasicBlock *BB) {
  // Initialize the collections. We will make a single pass over the block.
  Stores.clear();
  GEPs.clear();

  // Visit the store and getelementptr instructions in BB and organize them in
  // Stores and GEPs according to the underlying objects of their pointer
  // operands.
  for (Instruction &I : *BB) {
    // Ignore store instructions that are volatile or have a pointer operand
    // that doesn't point to a scalar type.
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (!SI->isSimple())
        continue;
      if (!isValidElementType(SI->getValueOperand()->getType()))
        continue;
      Stores[getUnderlyingObject(SI->getPointerOperand())].push_back(SI);
    }

    // Ignore getelementptr instructions that have more than one index, a
    // constant index, or a pointer operand that doesn't point to a scalar
    // type.
    else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      if (GEP->getNumIndices() != 1)
        continue;
      Value *Idx = GEP->idx_begin()->get();
      if (isa<Constant>(Idx))
        continue;
      if (!isValidElementType(Idx->getType()))
        continue;
      if (GEP->getType()->isVectorTy())
        continue;
      GEPs[GEP->getPointerOperand()].push_back(GEP);
    }
  }
}

template <typename T>
static bool tryToVectorizeSequence(
    SmallVectorImpl<T *> &Incoming, function_ref<bool(T *, T *)> Comparator,
    function_ref<bool(T *, T *)> AreCompatible,
    function_ref<bool(ArrayRef<T *>, bool)> TryToVectorizeHelper,
    bool MaxVFOnly, BoUpSLP &R) {
  bool Changed = false;
  // Sort by type, parent, operands.
  stable_sort(Incoming, Comparator);

  // Try to vectorize elements base on their type.
  SmallVector<T *> Candidates;
  SmallVector<T *> VL;
  for (auto *IncIt = Incoming.begin(), *E = Incoming.end(); IncIt != E;
       VL.clear()) {
    // Look for the next elements with the same type, parent and operand
    // kinds.
    auto *I = dyn_cast<Instruction>(*IncIt);
    if (!I || R.isDeleted(I)) {
      ++IncIt;
      continue;
    }
    auto *SameTypeIt = IncIt;
    while (SameTypeIt != E && (!isa<Instruction>(*SameTypeIt) ||
                               R.isDeleted(cast<Instruction>(*SameTypeIt)) ||
                               AreCompatible(*SameTypeIt, *IncIt))) {
      auto *I = dyn_cast<Instruction>(*SameTypeIt);
      ++SameTypeIt;
      if (I && !R.isDeleted(I))
        VL.push_back(cast<T>(I));
    }

    // Try to vectorize them.
    unsigned NumElts = VL.size();
    LLVM_DEBUG(dbgs() << "SLP: Trying to vectorize starting at nodes ("
                      << NumElts << ")\n");
    // The vectorization is a 3-state attempt:
    // 1. Try to vectorize instructions with the same/alternate opcodes with the
    // size of maximal register at first.
    // 2. Try to vectorize remaining instructions with the same type, if
    // possible. This may result in the better vectorization results rather than
    // if we try just to vectorize instructions with the same/alternate opcodes.
    // 3. Final attempt to try to vectorize all instructions with the
    // same/alternate ops only, this may result in some extra final
    // vectorization.
    if (NumElts > 1 && TryToVectorizeHelper(ArrayRef(VL), MaxVFOnly)) {
      // Success start over because instructions might have been changed.
      Changed = true;
      VL.swap(Candidates);
      Candidates.clear();
      for (T *V : VL) {
        if (auto *I = dyn_cast<Instruction>(V); I && !R.isDeleted(I))
          Candidates.push_back(V);
      }
    } else {
      /// \Returns the minimum number of elements that we will attempt to
      /// vectorize.
      auto GetMinNumElements = [&R](Value *V) {
        unsigned EltSize = R.getVectorElementSize(V);
        return std::max(2U, R.getMaxVecRegSize() / EltSize);
      };
      if (NumElts < GetMinNumElements(*IncIt) &&
          (Candidates.empty() ||
           Candidates.front()->getType() == (*IncIt)->getType())) {
        for (T *V : VL) {
          if (auto *I = dyn_cast<Instruction>(V); I && !R.isDeleted(I))
            Candidates.push_back(V);
        }
      }
    }
    // Final attempt to vectorize instructions with the same types.
    if (Candidates.size() > 1 &&
        (SameTypeIt == E || (*SameTypeIt)->getType() != (*IncIt)->getType())) {
      if (TryToVectorizeHelper(Candidates, /*MaxVFOnly=*/false)) {
        // Success start over because instructions might have been changed.
        Changed = true;
      } else if (MaxVFOnly) {
        // Try to vectorize using small vectors.
        SmallVector<T *> VL;
        for (auto *It = Candidates.begin(), *End = Candidates.end(); It != End;
             VL.clear()) {
          auto *I = dyn_cast<Instruction>(*It);
          if (!I || R.isDeleted(I)) {
            ++It;
            continue;
          }
          auto *SameTypeIt = It;
          while (SameTypeIt != End &&
                 (!isa<Instruction>(*SameTypeIt) ||
                  R.isDeleted(cast<Instruction>(*SameTypeIt)) ||
                  AreCompatible(*SameTypeIt, *It))) {
            auto *I = dyn_cast<Instruction>(*SameTypeIt);
            ++SameTypeIt;
            if (I && !R.isDeleted(I))
              VL.push_back(cast<T>(I));
          }
          unsigned NumElts = VL.size();
          if (NumElts > 1 && TryToVectorizeHelper(ArrayRef(VL),
                                                  /*MaxVFOnly=*/false))
            Changed = true;
          It = SameTypeIt;
        }
      }
      Candidates.clear();
    }

    // Start over at the next instruction of a different type (or the end).
    IncIt = SameTypeIt;
  }
  return Changed;
}

bool SLPVectorizerPass::vectorizeStoreChains(BoUpSLP &R) {
  bool Changed = false;
  // Sort by type, base pointers and values operand. Value operands must be
  // compatible (have the same opcode, same parent), otherwise it is
  // definitely not profitable to try to vectorize them.
  auto &&StoreSorter = [this](StoreInst *V, StoreInst *V2) {
    if (V->getValueOperand()->getType()->getTypeID() <
        V2->getValueOperand()->getType()->getTypeID())
      return true;
    if (V->getValueOperand()->getType()->getTypeID() >
        V2->getValueOperand()->getType()->getTypeID())
      return false;
    if (V->getPointerOperandType()->getTypeID() <
        V2->getPointerOperandType()->getTypeID())
      return true;
    if (V->getPointerOperandType()->getTypeID() >
        V2->getPointerOperandType()->getTypeID())
      return false;
    if (V->getValueOperand()->getType()->getScalarSizeInBits() <
        V2->getValueOperand()->getType()->getScalarSizeInBits())
      return true;
    if (V->getValueOperand()->getType()->getScalarSizeInBits() >
        V2->getValueOperand()->getType()->getScalarSizeInBits())
      return false;
    // UndefValues are compatible with all other values.
    if (auto *I1 = dyn_cast<Instruction>(V->getValueOperand()))
      if (auto *I2 = dyn_cast<Instruction>(V2->getValueOperand())) {
        DomTreeNodeBase<llvm::BasicBlock> *NodeI1 =
            DT->getNode(I1->getParent());
        DomTreeNodeBase<llvm::BasicBlock> *NodeI2 =
            DT->getNode(I2->getParent());
        assert(NodeI1 && "Should only process reachable instructions");
        assert(NodeI2 && "Should only process reachable instructions");
        assert((NodeI1 == NodeI2) ==
                   (NodeI1->getDFSNumIn() == NodeI2->getDFSNumIn()) &&
               "Different nodes should have different DFS numbers");
        if (NodeI1 != NodeI2)
          return NodeI1->getDFSNumIn() < NodeI2->getDFSNumIn();
        return I1->getOpcode() < I2->getOpcode();
      }
    return V->getValueOperand()->getValueID() <
           V2->getValueOperand()->getValueID();
  };

  auto &&AreCompatibleStores = [this](StoreInst *V1, StoreInst *V2) {
    if (V1 == V2)
      return true;
    if (V1->getValueOperand()->getType() != V2->getValueOperand()->getType())
      return false;
    if (V1->getPointerOperandType() != V2->getPointerOperandType())
      return false;
    // Undefs are compatible with any other value.
    if (isa<UndefValue>(V1->getValueOperand()) ||
        isa<UndefValue>(V2->getValueOperand()))
      return true;
    if (auto *I1 = dyn_cast<Instruction>(V1->getValueOperand()))
      if (auto *I2 = dyn_cast<Instruction>(V2->getValueOperand())) {
        if (I1->getParent() != I2->getParent())
          return false;
        InstructionsState S = getSameOpcode({I1, I2}, *TLI);
        return S.getOpcode() > 0;
      }
    if (isa<Constant>(V1->getValueOperand()) &&
        isa<Constant>(V2->getValueOperand()))
      return true;
    return V1->getValueOperand()->getValueID() ==
           V2->getValueOperand()->getValueID();
  };

  // Attempt to sort and vectorize each of the store-groups.
  DenseSet<std::tuple<Value *, Value *, Value *, Value *, unsigned>> Attempted;
  for (auto &Pair : Stores) {
    if (Pair.second.size() < 2)
      continue;

    // Reverse stores to do bottom-to-top analysis. This is important if the
    // values are stores to the same addresses several times, in this case need
    // to follow the stores order (reversed to meet the memory dependecies).
    SmallVector<StoreInst *> ReversedStores(Pair.second.rbegin(),
                                            Pair.second.rend());
    Changed |= tryToVectorizeSequence<StoreInst>(
        ReversedStores, StoreSorter, AreCompatibleStores,
        [&](ArrayRef<StoreInst *> Candidates, bool) {
          return vectorizeStores(Candidates, R, Attempted);
        },
        /*MaxVFOnly=*/false, R);
  }
  return Changed;
}
