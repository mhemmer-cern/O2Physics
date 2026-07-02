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

/// \file emcalPhotonCutsTest.cxx
/// \brief Task to set the EMCal photon cuts which will be used by other tasks
/// \author M. Hemmer, marvin.hemmer@cern.ch

#include "PWGEM/PhotonMeson/Core/EMCPhotonCut.h"
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
#include <unordered_map>
#include <vector>

using namespace o2;
using namespace o2::aod;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::soa;
using namespace o2::aod::pwgem::photon;

struct EmcalPhotonCutsTest {

  Configurable<std::string> cutDeviceName{"cutDeviceName", "emcal-photon-cuts", "Name of the device from which the configs are to be taken"};
  struct : ConfigurableGroup {
    std::string prefix = "emccuts";
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

  EMCPhotonCut fEMCCut;

  // One pass over device.options -> name/value map, built once per producer device.
  std::unordered_map<std::string, o2::framework::ConfigParamSpec const*> fOptionMap;

  void buildOptionMap(o2::framework::DeviceSpec const& device)
  {
    fOptionMap.clear();
    fOptionMap.reserve(device.options.size());
    for (auto const& option : device.options) {
      fOptionMap.emplace(option.name, &option);
    }
  }

  template <typename T>
  T getOption(std::string const& name) const
  {
    auto it = fOptionMap.find(name);
    if (it == fOptionMap.end()) {
      LOG(fatal) << "EmcalPhotonCutsTest: option " << name << " not found on producer device. Config name mismatch, or producer task not in workflow?";
    }
    return it->second->defaultValue.get<T>();
  }

  void defineEMCCut()
  {
    fEMCCut = EMCPhotonCut("fEMCCut", "fEMCCut");

    fEMCCut.SetClusterizer(getOption<std::string>("emccuts.clusterDefinition"));

    auto tmEta = getOption<std::vector<float>>("emccuts.cfgEMCTMEta");
    auto tmPhi = getOption<std::vector<float>>("emccuts.cfgEMCTMPhi");
    fEMCCut.SetTrackMatchingEtaParams(tmEta.at(0), tmEta.at(1), tmEta.at(2));
    fEMCCut.SetTrackMatchingPhiParams(tmPhi.at(0), tmPhi.at(1), tmPhi.at(2));

    auto secTmEta = getOption<std::vector<float>>("emccuts.emcSecTMEta");
    auto secTmPhi = getOption<std::vector<float>>("emccuts.emcSecTMPhi");
    fEMCCut.SetSecTrackMatchingEtaParams(secTmEta.at(0), secTmEta.at(1), secTmEta.at(2));
    fEMCCut.SetSecTrackMatchingPhiParams(secTmPhi.at(0), secTmPhi.at(1), secTmPhi.at(2));

    fEMCCut.SetMinEoverP(getOption<float>("emccuts.cfgEMCEoverp"));
    fEMCCut.SetMinE(getOption<float>("emccuts.cfgEMCminE"));
    fEMCCut.SetMinNCell(getOption<int>("emccuts.cfgEMCminNCell"));
    fEMCCut.SetM02Range(getOption<float>("emccuts.cfgEMCminM02"), getOption<float>("emccuts.cfgEMCmaxM02"));
    fEMCCut.SetTimeRange(getOption<float>("emccuts.cfgEMCminTime"), getOption<float>("emccuts.cfgEMCmaxTime"));
    fEMCCut.SetUseExoticCut(getOption<bool>("emccuts.cfgEMCUseExoticCut"));
    fEMCCut.SetUseTM(getOption<bool>("emccuts.cfgEMCUseTM"));
    fEMCCut.SetUseSecondaryTM(getOption<bool>("emccuts.emcUseSecondaryTM"));
    fEMCCut.SetDoQA(getOption<bool>("emccuts.cfgEnableQA"));
  }

  void init(InitContext& context)
  {
    auto& workflows = context.services().get<o2::framework::RunningWorkflowInfo const>();
    bool found = false;
    for (o2::framework::DeviceSpec const& device : workflows.devices) {
      if (device.name.compare(cutDeviceName.value) == 0) {
        buildOptionMap(device);
        defineEMCCut();
        found = true;
        break; // device names are unique in a workflow, no need to keep scanning
      }
    }
    if (!found) {
      LOG(fatal) << "EmcalPhotonCutsTest: producer device " << cutDeviceName.value << " not found in workflow.";
    }
  }; // end init

  // Pi0 from EMCal
  void process(aod::MinClusters const& /*clusters*/)
  {
    LOG(info) << "Runnig process!";
    return;
  }
}; // End struct EmcalPhotonCutsTest

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{adaptAnalysisTask<EmcalPhotonCutsTest>(cfgc)};
}
