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

/// \file emcalPhotonCuts.cxx
/// \brief Task to set the EMCal photon cuts which will be used by other tasks
/// \author M. Hemmer, marvin.hemmer@cern.ch

#include "PWGEM/PhotonMeson/DataModel/EventTables.h"
#include "PWGEM/PhotonMeson/DataModel/GammaTablesRedux.h"
#include "PWGEM/PhotonMeson/DataModel/gammaTables.h"

#include <CCDB/BasicCCDBManager.h>
#include <CommonConstants/MathConstants.h>
#include <EMCALBase/Geometry.h>
#include <EMCALBase/GeometryBase.h>
#include <EMCALCalib/BadChannelMap.h>
#include <Framework/ASoA.h>
#include <Framework/ASoAHelpers.h>
#include <Framework/AnalysisDataModel.h>
#include <Framework/AnalysisHelpers.h>
#include <Framework/AnalysisTask.h>
#include <Framework/BinningPolicy.h>
#include <Framework/Configurable.h>
#include <Framework/Expressions.h>
#include <Framework/GroupedCombinations.h>
#include <Framework/InitContext.h>
#include <Framework/runDataProcessing.h>

#include <cmath>
#include <string>
#include <vector>

using namespace o2;
using namespace o2::aod;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::soa;
using namespace o2::aod::pwgem::photon;

struct EmcalPhotonCuts {

  struct : ConfigurableGroup {
    std::string prefix = "emccuts";
    Configurable<std::string> clusterDefinition{"clusterDefinition", "kV3MostSplitSmallestTimeDiff", "Clusterizer to be selected, e.g. V3Default"};
    Configurable<float> cfgEMCminTime{"cfgEMCminTime", -25., "Minimum cluster time for EMCal time cut"};
    Configurable<float> cfgEMCmaxTime{"cfgEMCmaxTime", +30., "Maximum cluster time for EMCal time cut"};
    Configurable<float> cfgEMCminM02{"cfgEMCminM02", 0.1, "Minimum M02 for EMCal M02 cut"};
    Configurable<float> cfgEMCmaxM02{"cfgEMCmaxM02", 0.7, "Maximum M02 for EMCal M02 cut"};
    Configurable<float> cfgEMCminE{"cfgEMCminE", 0.7, "Minimum cluster energy for EMCal energy cut"};
    Configurable<int> cfgEMCminNCell{"cfgEMCminNCell", 1, "Minimum number of cells per cluster for EMCal NCell cut"};
    Configurable<std::vector<float>> cfgEMCTMEta{"cfgEMCTMEta", {0.01f, 4.07f, -2.5f}, "|eta| <= [0]+(pT+[1])^[2] for EMCal track matching"};
    Configurable<std::vector<float>> cfgEMCTMPhi{"cfgEMCTMPhi", {0.015f, 3.65f, -2.f}, "|phi| <= [0]+(pT+[1])^[2] for EMCal track matching"};
    Configurable<std::vector<float>> emcSecTMEta{"emcSecTMEta", {0.01f, 4.07f, -2.5f}, "|eta| <= [0]+(pT+[1])^[2] for EMCal track matching"};
    Configurable<std::vector<float>> emcSecTMPhi{"emcSecTMPhi", {0.015f, 3.65f, -2.f}, "|phi| <= [0]+(pT+[1])^[2] for EMCal track matching"};
    Configurable<float> cfgEMCEoverp{"cfgEMCEoverp", 1.75, "Minimum cluster energy over track momentum for EMCal track matching"};
    Configurable<bool> cfgEMCUseExoticCut{"cfgEMCUseExoticCut", true, "FLag to use the EMCal exotic cluster cut"};
    Configurable<bool> cfgEMCUseTM{"cfgEMCUseTM", false, "flag to use EMCal track matching cut or not"};
    Configurable<bool> emcUseSecondaryTM{"emcUseSecondaryTM", false, "flag to use EMCal secondary track matching cut or not"};
    Configurable<bool> cfgEnableQA{"cfgEnableQA", false, "flag to turn QA plots on/off"};
  } emccuts;

  void init(InitContext&)
  {
  } // end init

  // Pi0 from EMCal
  void process(aod::MinClusters const& /*clusters*/)
  {
    LOG(info) << "Runnig process!";
    return;
  }
}; // End struct EmcalPhotonCuts

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{adaptAnalysisTask<EmcalPhotonCuts>(cfgc)};
}
