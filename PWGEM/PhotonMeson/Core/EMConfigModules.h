// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file EMConfigModules.h
/// \brief Shared configurable groups for PWGEM/PhotonMeson tasks and a helper to inherit them from a central config device (em-config-holder)
/// \author Marvin Hemmer (marvin.hemmer@cern.ch) - Goethe University Frankfurt

#ifndef PWGEM_PHOTONMESON_CORE_EMCONFIGMODULES_H_
#define PWGEM_PHOTONMESON_CORE_EMCONFIGMODULES_H_

#include "PWGEM/PhotonMeson/Core/EMCPhotonCut.h"
#include "PWGEM/PhotonMeson/Core/EMPhotonEventCut.h"

#include <Framework/ConfigParamSpec.h>
#include <Framework/Configurable.h>
#include <Framework/DeviceSpec.h>
#include <Framework/InitContext.h>
#include <Framework/Logger.h>
#include <Framework/RunningWorkflowInfo.h>
#include <Framework/StructToTuple.h>
#include <Framework/Traits.h>
#include <Framework/Variant.h>

#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace o2::aod::pwgem::pm::config
{

// ---------------------------------------------------------------------------
// Shared configurable groups
// Names and prefixes must stay identical between producer (em-config-holder)
// and consumers, since options are matched by their full name "prefix.name".
// ---------------------------------------------------------------------------

struct EventCutConfigurables : o2::framework::ConfigurableGroup {
  std::string prefix = "eventcut_group";
  o2::framework::Configurable<float> cfgZvtxMax{"cfgZvtxMax", 10.f, "max. Zvtx"};
  o2::framework::Configurable<bool> cfgRequireSel8{"cfgRequireSel8", true, "require sel8 in event cut"};
  o2::framework::Configurable<bool> cfgRequireFT0AND{"cfgRequireFT0AND", true, "require FT0AND in event cut"};
  o2::framework::Configurable<bool> cfgRequireNoTFB{"cfgRequireNoTFB", false, "require No time frame border in event cut"};
  o2::framework::Configurable<bool> cfgRequireNoITSROFB{"cfgRequireNoITSROFB", false, "require no ITS readout frame border in event cut"};
  o2::framework::Configurable<bool> cfgRequireNoSameBunchPileup{"cfgRequireNoSameBunchPileup", false, "require no same bunch pileup in event cut"};
  o2::framework::Configurable<bool> cfgRequireVertexITSTPC{"cfgRequireVertexITSTPC", false, "require Vertex ITSTPC in event cut"}; // ITS-TPC matched track contributes PV.
  o2::framework::Configurable<bool> cfgRequireGoodZvtxFT0vsPV{"cfgRequireGoodZvtxFT0vsPV", false, "require good Zvtx between FT0 vs. PV in event cut"};
  o2::framework::Configurable<bool> cfgRequireEMCReadoutInMB{"cfgRequireEMCReadoutInMB", false, "require the EMC to be read out in an MB collision (kTVXinEMC)"};
  o2::framework::Configurable<bool> cfgRequireEMCHardwareTriggered{"cfgRequireEMCHardwareTriggered", false, "require the EMC to be hardware triggered (kEMC7 or kDMC7)"};
  o2::framework::Configurable<int> cfgOccupancyMin{"cfgOccupancyMin", -1, "min. occupancy"};
  o2::framework::Configurable<int> cfgOccupancyMax{"cfgOccupancyMax", 1000000000, "max. occupancy"};
  o2::framework::Configurable<bool> onlyKeepWeightedEvents{"onlyKeepWeightedEvents", false, "flag to keep only weighted events (for JJ MCs) and remove all MB events (with weight = 1)"};
};

struct EMCCutConfigurables : o2::framework::ConfigurableGroup {
  std::string prefix = "emccut_group";
  o2::framework::Configurable<std::string> clusterDefinition{"clusterDefinition", "kV3Default", "Clusterizer to be selected, e.g. V3Default"};
  o2::framework::Configurable<float> minOpenAngle{"minOpenAngle", 0.0202, "apply min opening angle"};
  o2::framework::Configurable<float> minClusterTime{"minClusterTime", -20., "Minimum cluster time for EMCal time cut"};
  o2::framework::Configurable<float> maxClusterTime{"maxClusterTime", +25., "Maximum cluster time for EMCal time cut"};
  o2::framework::Configurable<float> minM02{"minM02", 0.1, "Minimum M02 for EMCal M02 cut"};
  o2::framework::Configurable<float> maxM02{"maxM02", 0.7, "Maximum M02 for EMCal M02 cut"};
  o2::framework::Configurable<float> minClusterE{"minClusterE", 0.7, "Minimum cluster energy for EMCal energy cut"};
  o2::framework::Configurable<int> minNCell{"minNCell", 1, "Minimum number of cells per cluster for EMCal NCell cut"};
  o2::framework::Configurable<std::vector<float>> cfgEMCTMEta{"cfgEMCTMEta", {0.01f, 4.07f, -2.5f}, "|eta| <= [0]+(pT+[1])^[2] for EMCal track matching"};
  o2::framework::Configurable<std::vector<float>> cfgEMCTMPhi{"cfgEMCTMPhi", {0.015f, 3.65f, -2.f}, "|phi| <= [0]+(pT+[1])^[2] for EMCal track matching"};
  o2::framework::Configurable<std::vector<float>> emcSecTMEta{"emcSecTMEta", {0.01f, 4.07f, -2.5f}, "|eta| <= [0]+(pT+[1])^[2] for EMCal track matching"};
  o2::framework::Configurable<std::vector<float>> emcSecTMPhi{"emcSecTMPhi", {0.015f, 3.65f, -2.f}, "|phi| <= [0]+(pT+[1])^[2] for EMCal track matching"};
  o2::framework::Configurable<float> tmEoverP{"tmEoverP", 1.75, "Minimum cluster energy over track momentum for EMCal track matching"};
  o2::framework::Configurable<bool> useExoticCut{"useExoticCut", true, "FLag to use the EMCal exotic cluster cut"};
  o2::framework::Configurable<bool> cfgEMCUseTM{"cfgEMCUseTM", false, "flag to use EMCal track matching cut or not"};
  o2::framework::Configurable<bool> emcUseSecondaryTM{"emcUseSecondaryTM", false, "flag to use EMCal secondary track matching cut or not"};
};

// ---------------------------------------------------------------------------
// Helpers to inherit configurable values from another device in the workflow
// ---------------------------------------------------------------------------

/// Assign the value stored in a DPL Variant to a configurable value.
/// Arrays are stored in the Variant as pointer + size, so they need special treatment.
/// \return false if the type is not supported (value left untouched)
template <typename T>
bool assignFromVariant(T& value, o2::framework::Variant const& variant)
{
  if constexpr (std::is_same_v<T, std::string>) {
    value = variant.get<const char*>();
  } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
    const std::string* ptr = variant.get<std::string*>();
    value.assign(ptr, ptr + variant.size());
  } else if constexpr (o2::framework::is_specialization_v<T, std::vector>) {
    using ElemT = typename T::value_type;
    const ElemT* ptr = variant.get<ElemT*>();
    value.assign(ptr, ptr + variant.size());
  } else if constexpr (o2::framework::variant_trait_v<T> != o2::framework::VariantType::Unknown) {
    value = variant.get<T>(); // scalars and LabeledArray<T>
  } else {
    return false; // e.g. ROOT-serialised (ptree) configurables
  }
  return true;
}

/// Fill all configurables of a ConfigurableGroup from the options of another device in the running workflow.
/// Must be called in init(), after the framework has filled the group (option names then already carry the "prefix." part).
/// \param context InitContext of the calling task
/// \param deviceName full device name of the config holder, including a possible workflow suffix (e.g. "em-config-holder_PCM")
/// \param group configurable group to be overwritten
/// \return true if the device was found and the group was updated, false otherwise (group keeps its local values)
template <typename TGroup>
  requires std::derived_from<TGroup, o2::framework::ConfigurableGroup>
bool readGroupFromDevice(o2::framework::InitContext& context, std::string const& deviceName, TGroup& group)
{
  if (deviceName.empty()) {
    return false;
  }

  auto const& workflow = context.services().template get<o2::framework::RunningWorkflowInfo const>();
  o2::framework::DeviceSpec const* configDevice = nullptr;
  for (auto const& device : workflow.devices) {
    if (device.name == deviceName) {
      configDevice = &device;
      break;
    }
  }
  if (configDevice == nullptr) {
    LOG(warn) << "Config device '" << deviceName << "' not found in workflow. Using local configurables.";
    return false;
  }

  std::unordered_map<std::string, o2::framework::Variant const*> options;
  options.reserve(configDevice->options.size());
  for (auto const& option : configDevice->options) {
    options.emplace(option.name, &option.defaultValue);
  }

  o2::framework::homogeneous_apply_refs<true>([&](auto& member) {
    if constexpr (requires { member.name; member.value; }) {
      auto it = options.find(member.name);
      if (it == options.end()) {
        LOG(warn) << "Option '" << member.name << "' not found in config device '" << deviceName << "'. Keeping local value.";
      } else if (!assignFromVariant(member.value, *it->second)) {
        LOG(warn) << "Option '" << member.name << "' has an unsupported type. Keeping local value.";
      }
    }
    return true;
  },
                                              group);

  LOG(info) << "Configurable group '" << group.prefix << "' taken from device '" << deviceName << "'";
  return true;
}

// ---------------------------------------------------------------------------
// Helpers to configure cut objects from the shared groups
// ---------------------------------------------------------------------------

inline void configureEventCut(EMPhotonEventCut& cut, EventCutConfigurables const& cfg)
{
  cut.SetRequireSel8(cfg.cfgRequireSel8.value);
  cut.SetRequireFT0AND(cfg.cfgRequireFT0AND.value);
  cut.SetZvtxRange(-cfg.cfgZvtxMax.value, +cfg.cfgZvtxMax.value);
  cut.SetRequireNoTFB(cfg.cfgRequireNoTFB.value);
  cut.SetRequireNoITSROFB(cfg.cfgRequireNoITSROFB.value);
  cut.SetRequireNoSameBunchPileup(cfg.cfgRequireNoSameBunchPileup.value);
  cut.SetRequireVertexITSTPC(cfg.cfgRequireVertexITSTPC.value);
  cut.SetRequireGoodZvtxFT0vsPV(cfg.cfgRequireGoodZvtxFT0vsPV.value);
  cut.SetRequireEMCReadoutInMB(cfg.cfgRequireEMCReadoutInMB.value);
  cut.SetRequireEMCHardwareTriggered(cfg.cfgRequireEMCHardwareTriggered.value);
  cut.SetTrackOccupancyRange(cfg.cfgOccupancyMin.value, cfg.cfgOccupancyMax.value);
}

inline void configureEMCCut(EMCPhotonCut& cut, EMCCutConfigurables const& cfg)
{
  cut.SetClusterizer(cfg.clusterDefinition.value);
  cut.SetMinE(cfg.minClusterE.value);
  cut.SetMinNCell(cfg.minNCell.value);
  cut.SetM02Range(cfg.minM02.value, cfg.maxM02.value);
  cut.SetTimeRange(cfg.minClusterTime.value, cfg.maxClusterTime.value);

  cut.SetUseExoticCut(cfg.useExoticCut.value);

  cut.SetTrackMatchingEtaParams(cfg.cfgEMCTMEta.value.at(0), cfg.cfgEMCTMEta.value.at(1), cfg.cfgEMCTMEta.value.at(2));
  cut.SetTrackMatchingPhiParams(cfg.cfgEMCTMPhi.value.at(0), cfg.cfgEMCTMPhi.value.at(1), cfg.cfgEMCTMPhi.value.at(2));
  cut.SetMinEoverP(cfg.tmEoverP.value);
  cut.SetUseTM(cfg.cfgEMCUseTM.value); // disables or enables TM

  cut.SetSecTrackMatchingEtaParams(cfg.emcSecTMEta.value.at(0), cfg.emcSecTMEta.value.at(1), cfg.emcSecTMEta.value.at(2));
  cut.SetSecTrackMatchingPhiParams(cfg.emcSecTMPhi.value.at(0), cfg.emcSecTMPhi.value.at(1), cfg.emcSecTMPhi.value.at(2));
  cut.SetUseSecondaryTM(cfg.emcUseSecondaryTM.value); // disables or enables secondary TM
}

} // namespace o2::aod::pwgem::pm::config

#endif // PWGEM_PHOTONMESON_CORE_EMCONFIGMODULES_H_
