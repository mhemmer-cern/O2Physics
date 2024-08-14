// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <climits>
#include <cstdlib>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>


#include "Framework/ASoAHelpers.h"
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoA.h"
#include "Framework/HistogramRegistry.h"

#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Centrality.h"

#include "PWGEM/PhotonMeson/DataModel/gammaTables.h"
#include "EMCALBase/Geometry.h"
#include "PWGJE/DataModel/EMCALClusters.h"
#include "PWGJE/DataModel/EMCALMatchedCollisions.h"
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/Constants.h"
#include "DataFormatsEMCAL/AnalysisCluster.h"

#include "CommonDataFormat/InteractionRecord.h"

#include "Math/Vector4D.h"

// \struct Run2Pi0EtaToGammaGamma
/// \brief Simple task for Run2 analysis in O2Physics
/// \author Marvin Hemmer <marvin.hemmer@cern.ch>, Goethe University Frankfurt
/// \since 01.08.2024
///
/// This task is meant to be used for QC for new converted run2 data

using namespace o2;
using namespace o2::aod;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::soa;
// using SelectedV0s = o2::soa::Filtered<o2::aod::V0Otfs>;

struct Run2Pi0EtaToGammaGamma {

  SliceCache cache;
  std::vector<double> zBins{VARIABLE_WIDTH, -10, -6, -2, +2, +6, +10};
  using BinningType = ColumnBinningPolicy<aod::collision::PosZ>;
  BinningType binningOnPositions{{zBins}, true};                                    // true is for 'ignore overflows' (true by default)
  SameKindPair<aod::Collisions, o2::aod::V0Otfs, BinningType> pair{binningOnPositions, 10, -1, &cache}; // indicates that 5 events should be mixed and under/overflow (-1) to be ignored

  HistogramRegistry mHistManager{"Run2Pi0EtaToGammaGammaHistograms"};

  ConfigurableAxis pTBinning{"pTBinning", {500, 0.0f, 50.0f}, "Binning used along pT axis for inv mass histograms"};
  ConfigurableAxis invmassBinning{"invmassBinning", {100, 0.0f, 0.8f}, "Binning used for inv mass axis in inv mass - pT histograms"};
  ConfigurableAxis count{"Nv0", {100, 0., 100}, "Binning used for counting particles"};
  ConfigurableAxis cuts{"Cuts", {5, -0.5, 4.5}, "Binning used for counting particles"};

  /// \brief Create output histograms and initialize geometry
  void init(InitContext const&)
  {
    // create histograms
    using o2HistType = HistType;
    // using o2Axis = AxisSpec;

    // meson related histograms
    mHistManager.add("NV0s", "Number of V0s", o2HistType::kTH1I, {count});
    mHistManager.add("NCuts", "Cuts", o2HistType::kTH1I, {cuts});
    mHistManager.add("invMassVsPt", "invariant mass and pT of meson candidates", o2HistType::kTH2F, {invmassBinning, pTBinning});
    mHistManager.add("invMassVsPtBackground", "invariant mass and pT of background meson candidates", o2HistType::kTH2F, {invmassBinning, pTBinning});
    auto hCuts = mHistManager.get<TH1>(HIST("NCuts"));
    hCuts->GetXaxis()->SetBinLabel(1, "V0s in");
    hCuts->GetXaxis()->SetBinLabel(2, "#psi_{pair}");
    hCuts->GetXaxis()->SetBinLabel(3, "cos(P)");
    hCuts->GetXaxis()->SetBinLabel(4, "#it{q}_{T}");
    hCuts->GetXaxis()->SetBinLabel(5, "V0s out");
  }

  Preslice<o2::aod::V0Otfs> perCollision_V0 = o2::aod::v0otf::collisionId;

  // Filter V0Filter = ((o2::aod::v0otf::chi2NDF < 4.0f) && (o2::aod::v0otf::alpha < 1.f));
  /// \brief Process EMCAL clusters that are matched to a collisions
  void processCollision(o2::aod::Collisions const& collisions, o2::aod::V0Otfs const& v0s)
  {
    // Same Event
    for (auto& collision : collisions) {
      auto v0s_per_collision = v0s.sliceBy(perCollision_V0, collision.globalIndex());
      mHistManager.fill(HIST("NV0s"), v0s_per_collision.size());
      for (auto& [g1, g2] : o2::soa::combinations(CombinationsStrictlyUpperIndexPolicy(v0s_per_collision, v0s_per_collision))) {
        mHistManager.fill(HIST("NCuts"), 0);
        if( (std::abs(g1.psiPair()) >= 0.18*exp(-0.055*g1.chi2NDF()) ) && (std::fabs(g2.psiPair()) >= 0.18*exp(-0.055*g2.chi2NDF()) )){
          mHistManager.fill(HIST("NCuts"), 1);
          continue;
        }
        if( (GetCosineOfPointingAngle(g1, collision) <= 0.85) && (GetCosineOfPointingAngle(g2, collision) <= 0.85)){
          mHistManager.fill(HIST("NCuts"), 2);
          continue;
        }
        ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float>>  v1(g1.px(), g1.py(), g1.pz(), g1.e());
        ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float>>  v2(g2.px(), g2.py(), g2.px(), g2.e());
        if( (g1.qt() >= 0.125 * v1.Pt()) && (g2.qt() >= 0.125 * v2.Pt()) ){
          mHistManager.fill(HIST("NCuts"), 3);
          continue;
        }
        auto v12 = v1 + v2;
        mHistManager.fill(HIST("invMassVsPt"), v12.M(), v12.Pt());
        mHistManager.fill(HIST("NCuts"), 4);
      }
    }

    //-------------------------------------------------------------------------
    // Mixed Event
    for (auto& [c1, v01, c2, v02] : pair) {
      for (auto& [t1, t2] : combinations(CombinationsFullIndexPolicy(v01, v02))) {
        if( (std::abs(t1.psiPair()) >= 0.18*exp(-0.055*t1.chi2NDF()) ) && (std::fabs(t2.psiPair()) >= 0.18*exp(-0.055*t2.chi2NDF()) )){
          continue;
        }
        if( (GetCosineOfPointingAngle(t1, c1) <= 0.85) && (GetCosineOfPointingAngle(t2, c2) <= 0.85) ){
          continue;
        }
        ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float>>  v1(t1.px(), t1.py(), t1.pz(), t1.e());
        ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float>>  v2(t2.px(), t2.py(), t2.px(), t2.e());
        if( (t1.qt() >= 0.125 * v1.Pt()) && (t2.qt() >= 0.125 * v2.Pt()) ){
          continue;
        }
        auto v12 = v1 + v2;
        mHistManager.fill(HIST("invMassVsPtBackground"), v12.M(), v12.Pt());
      }
    }
  }
  PROCESS_SWITCH(Run2Pi0EtaToGammaGamma, processCollision, "Process clusters from collision", true);

  double GetCosineOfPointingAngle( o2::aod::V0Otfs::iterator const& photon, o2::aod::Collisions::iterator const& collision) const{
  // calculates the pointing angle of the recalculated V0

  double momV0[3] = {photon.px(),photon.py(),photon.pz()};
  double PosV0[3] = { photon.cx() - collision.posX(),
                      photon.cy() - collision.posY(),
                      photon.cz() - collision.posZ() }; //Recalculated V0 Position vector

  double momV02 = momV0[0]*momV0[0] + momV0[1]*momV0[1] + momV0[2]*momV0[2];
  double PosV02 = PosV0[0]*PosV0[0] + PosV0[1]*PosV0[1] + PosV0[2]*PosV0[2];


  double cosinePointingAngle = -999;
  if(momV02*PosV02 > 0.0)
    cosinePointingAngle = (PosV0[0]*momV0[0] +  PosV0[1]*momV0[1] + PosV0[2]*momV0[2] ) / TMath::Sqrt(momV02 * PosV02);

  return cosinePointingAngle;
}
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<Run2Pi0EtaToGammaGamma>(cfgc, TaskName{"EMCPi0QCTask"},SetDefaultProcesses{{{"processCollision", true}}})
  };
}
