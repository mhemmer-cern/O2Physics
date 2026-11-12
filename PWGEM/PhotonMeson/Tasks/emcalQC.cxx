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

/// \file emcalQC.cxx
/// \brief Task that runs basic EMCal cluster QA for derived data in the EM format
/// \author Nicolas Strangmann (nicolas.strangmann@cern.ch) Goethe University Frankfurt
///

#include "PWGEM/PhotonMeson/Core/EMCPhotonCut.h"
#include "PWGEM/PhotonMeson/Core/EMConfigModules.h"
#include "PWGEM/PhotonMeson/Core/EMPhotonEventCut.h"
#include "PWGEM/PhotonMeson/DataModel/EventTables.h"
#include "PWGEM/PhotonMeson/DataModel/GammaTablesRedux.h"
#include "PWGEM/PhotonMeson/DataModel/gammaTables.h"
#include "PWGEM/PhotonMeson/Utils/ClusterHistograms.h"

#include "Common/CCDB/EventSelectionParams.h"
#include "Common/CCDB/TriggerAliases.h"

#include <Framework/ASoA.h>
#include <Framework/AnalysisDataModel.h>
#include <Framework/AnalysisTask.h>
#include <Framework/Concepts.h>
#include <Framework/Configurable.h>
#include <Framework/HistogramRegistry.h>
#include <Framework/HistogramSpec.h>
#include <Framework/InitContext.h>
#include <Framework/OutputObjHeader.h>
#include <Framework/runDataProcessing.h>

#include <TH1.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using namespace o2;
using namespace o2::aod;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::soa;
using namespace o2::aod::pwgem::photon;
using namespace o2::aod::pwgem::pm;

using MyCollisions = soa::Join<aod::PMEvents, aod::EMEventsAlias, aod::EMEventsMult_000, aod::EMEventsCent_000, aod::EMEventsQvec_001, aod::EMEventsWeight>;
using MyCollision = MyCollisions::iterator;

using EMCalPhotons = soa::Join<aod::EMCEMEventIds, aod::MinClusters>;
using EMCalPhoton = EMCalPhotons::iterator;

struct EmcalQC {

  Configurable<bool> cfgDo2DQA{"cfgDo2DQA", true, "perform 2 dimensional cluster QA"};
  Configurable<std::string> cfgConfigDeviceName{"cfgConfigDeviceName", "", "device name of em-config-holder (incl. suffix) to take eventcut_group and emccut_group from. Empty: use local values"};

  EMPhotonEventCut fEMEventCut;
  config::EventCutConfigurables eventcuts;

  EMCPhotonCut fEMCCut;
  config::EMCCutConfigurables emccuts;

  struct : ConfigurableGroup {
    std::string prefix = "axis_group";
    ConfigurableAxis thnConfigAxisE{"thnConfigAxisE", {100, 0., 20.}, "energy axis for photon candidates"};
    ConfigurableAxis thnConfigAxisPt{"thnConfigAxisPt", {100, 0., 20.}, "pT axis for matched tracks"};
  } axisGroup;

  void defineEMEventCut()
  {
    fEMEventCut = EMPhotonEventCut("fEMEventCut", "fEMEventCut");
    config::configureEventCut(fEMEventCut, eventcuts);
  }

  void defineEMCCut()
  {
    config::configureEMCCut(fEMCCut, emccuts);
  }

  HistogramRegistry fRegistry{"output", {}, OutputObjHandlingPolicy::AnalysisObject, false, false};

  SliceCache cache;

  PresliceOptional<MinMTracks> perEMCClusterMT = o2::aod::mintm::minClusterId;
  PresliceOptional<MinMSTracks> perEMCClusterMS = o2::aod::mintm::minClusterId;
  PresliceOptional<EMCalPhotons> perCollisionEMC = o2::aod::emccluster::pmeventId;

  void init(InitContext& context)
  {
    config::readGroupFromDevice(context, cfgConfigDeviceName.value, eventcuts);
    config::readGroupFromDevice(context, cfgConfigDeviceName.value, emccuts);

    defineEMCCut();
    defineEMEventCut();

    const AxisSpec thnAxisE{axisGroup.thnConfigAxisE, "#it{E}} (GeV)"};
    const AxisSpec thnAxisPt{axisGroup.thnConfigAxisPt, "#it{p}_{T} (GeV/#it{c})"};
    const AxisSpec thAxisdEta{200, -0.06, 0.06, "#Delta#eta"};
    const AxisSpec thAxisdPhi{200, -0.06, 0.06, "#Delta#varphi (rad)"};

    fEMEventCut.addEventHistograms(&fRegistry);
    auto hEMCCollisionCounter = fRegistry.add<TH1>("Event/hEMCCollisionCounter", "Number of collisions after event cuts", HistType::kTH1D, {{7, 0.5, 7.5}}, false);
    hEMCCollisionCounter->GetXaxis()->SetBinLabel(1, "all");
    hEMCCollisionCounter->GetXaxis()->SetBinLabel(2, "+TVX");         // TVX
    hEMCCollisionCounter->GetXaxis()->SetBinLabel(3, "+|z|<10cm");    // TVX with z < 10cm
    hEMCCollisionCounter->GetXaxis()->SetBinLabel(4, "+Sel8");        // TVX with z < 10cm and Sel8
    hEMCCollisionCounter->GetXaxis()->SetBinLabel(5, "+Good z vtx");  // TVX with z < 10cm and Sel8 and good z xertex
    hEMCCollisionCounter->GetXaxis()->SetBinLabel(6, "+unique");      // TVX with z < 10cm and Sel8 and good z xertex and unique (only collision in the BC)
    hEMCCollisionCounter->GetXaxis()->SetBinLabel(7, "+EMC readout"); // TVX with z < 10cm and Sel8 and good z xertex and unique (only collision in the BC) and kTVXinEMC
    o2::aod::pwgem::photonmeson::utils::clusterhistogram::addClusterHistograms(&fRegistry, cfgDo2DQA);

    if (doprocessQCTM) {
      fRegistry.add("Cluster/hDeltaEtaPhiAllPrimTracks", "#Delta#eta #Delta#varphi distribution of all matched prim. tracks", HistType::kTH3D, {thAxisdEta, thAxisdPhi, thnAxisE}, false);
      fRegistry.add("Cluster/hDeltaEtaPhiClosestPrimTracks", "#Delta#eta #Delta#varphi distribution of the Closest matched prim. track", HistType::kTH3D, {thAxisdEta, thAxisdPhi, thnAxisE}, false);
      fRegistry.add("Cluster/hDeltaEtaPhiAllSecTracks", "#Delta#eta #Delta#varphi distribution of all matched sec. tracks", HistType::kTH3D, {thAxisdEta, thAxisdPhi, thnAxisE}, false);
      fRegistry.add("Cluster/hDeltaEtaPhiClosestSecTracks", "#Delta#eta #Delta#varphi distribution of the Closest matched sec. track", HistType::kTH3D, {thAxisdEta, thAxisdPhi, thnAxisE}, false);

      fRegistry.add("Cluster/hDeltaEtaPhiAllPrimTracksTrackPt", "#Delta#eta #Delta#varphi distribution of all matched prim. tracks", HistType::kTH3D, {thAxisdEta, thAxisdPhi, thnAxisPt}, false);
      fRegistry.add("Cluster/hDeltaEtaPhiClosestPrimTracksTrackPt", "#Delta#eta #Delta#varphi distribution of the Closest matched prim. track", HistType::kTH3D, {thAxisdEta, thAxisdPhi, thnAxisPt}, false);
      fRegistry.add("Cluster/hDeltaEtaPhiAllSecTracksTrackPt", "#Delta#eta #Delta#varphi distribution of all matched sec. tracks", HistType::kTH3D, {thAxisdEta, thAxisdPhi, thnAxisPt}, false);
      fRegistry.add("Cluster/hDeltaEtaPhiClosestSecTracksTrackPt", "#Delta#eta #Delta#varphi distribution of the Closest matched sec. track", HistType::kTH3D, {thAxisdEta, thAxisdPhi, thnAxisPt}, false);
    }
  }

  bool isEventGood(MyCollision const& collision)
  {

    if (eventcuts.onlyKeepWeightedEvents && std::fabs(collision.weight() - 1.) < 1E-10) {
      return false;
    }

    fRegistry.fill(HIST("Event/hEMCCollisionCounter"), 1, collision.weight());
    if (!eventcuts.cfgRequireFT0AND || collision.selection_bit(o2::aod::evsel::kIsTriggerTVX)) {
      fRegistry.fill(HIST("Event/hEMCCollisionCounter"), 2, collision.weight());
      if (std::abs(collision.posZ()) < eventcuts.cfgZvtxMax) {
        fRegistry.fill(HIST("Event/hEMCCollisionCounter"), 3, collision.weight());
        if (!eventcuts.cfgRequireSel8 || collision.sel8()) {
          fRegistry.fill(HIST("Event/hEMCCollisionCounter"), 4, collision.weight());
          if (!eventcuts.cfgRequireGoodZvtxFT0vsPV || collision.selection_bit(o2::aod::evsel::kIsGoodZvtxFT0vsPV)) {
            fRegistry.fill(HIST("Event/hEMCCollisionCounter"), 5, collision.weight());
            if (!eventcuts.cfgRequireNoSameBunchPileup || collision.selection_bit(o2::aod::evsel::kNoSameBunchPileup)) {
              fRegistry.fill(HIST("Event/hEMCCollisionCounter"), 6, collision.weight());
              if (!eventcuts.cfgRequireEMCReadoutInMB || collision.alias_bit(kTVXinEMC)) {
                fRegistry.fill(HIST("Event/hEMCCollisionCounter"), 7, collision.weight());
              }
            }
          }
        }
      }
    }

    fEMEventCut.fillEventInfo<0>(&fRegistry, collision, collision.weight());
    if (!fEMEventCut.IsSelected(collision)) {
      return false;
    }

    fEMEventCut.fillEventInfo<1>(&fRegistry, collision, collision.weight());
    fEMEventCut.fillAccepted(&fRegistry, collision.weight());
    return true;
  }

  template <o2::soa::is_table TClusters>
  void doClusterQA(MyCollision const& collision, TClusters const& clusters)
  {
    fRegistry.fill(HIST("Cluster/before/hNgamma"), clusters.size(), collision.weight());
    int ngAfter = 0;
    for (const auto& cluster : clusters) {
      // Fill the cluster properties before applying any cuts
      if (!fEMCCut.IsSelectedEMCal(EMCPhotonCut::EMCPhotonCuts::kDefinition, cluster)) {
        continue;
      }
      o2::aod::pwgem::photonmeson::utils::clusterhistogram::fillClusterHistograms<0>(&fRegistry, cluster, cfgDo2DQA, collision.weight());

      // Apply cuts one by one and fill in hClusterQualityCuts histogram
      fRegistry.fill(HIST("Cluster/hClusterQualityCuts"), 0., cluster.e(), collision.weight());

      // Define two boleans to see, whether the cluster "survives" the EMC cluster cuts to later check, whether the cuts in this task align with the ones in EMCPhotonCut.h:
      bool survivesIsSelectedEMCalCuts = true; // Survives "manual" cuts listed in this task
      bool survivesIsSelectedCuts = true;

      survivesIsSelectedCuts = fEMCCut.IsSelected(cluster);
      for (int icut = 0; icut < static_cast<int>(EMCPhotonCut::EMCPhotonCuts::kNCuts); icut++) { // Loop through different cut observables
        auto specificcut = static_cast<EMCPhotonCut::EMCPhotonCuts>(icut);
        if (!fEMCCut.IsSelectedEMCal(specificcut, cluster)) { // Check whether cluster passes this cluster requirement, if not, fill why in the next row
          fRegistry.fill(HIST("Cluster/hClusterQualityCuts"), icut + 1, cluster.e(), collision.weight());
          survivesIsSelectedEMCalCuts = false;
        }
      }

      if (survivesIsSelectedCuts != survivesIsSelectedEMCalCuts) {
        LOGF(info, "Cummulative application of IsSelectedEMCal cuts does not equal the IsSelected result");
        LOGF(info, "Cummulative application of IsSelectedEMCal resulted in %d", survivesIsSelectedEMCalCuts);
        LOGF(info, "IsSelected resulted in %d", survivesIsSelectedCuts);
      }

      if (survivesIsSelectedCuts) {
        o2::aod::pwgem::photonmeson::utils::clusterhistogram::fillClusterHistograms<1>(&fRegistry, cluster, cfgDo2DQA, collision.weight());
        fRegistry.fill(HIST("Cluster/hClusterQualityCuts"), static_cast<double>(EMCPhotonCut::EMCPhotonCuts::kNCuts) + 1., cluster.e(), collision.weight());
        ++ngAfter;
      }
    }
    fRegistry.fill(HIST("Cluster/after/hNgamma"), ngAfter, collision.weight());
  }

  template <o2::soa::is_table TClusters, o2::soa::is_table TMatchedTracks, o2::soa::is_table TMatchedSecondaries>
  void doClusterQAWithTM(MyCollision const& collision, TClusters const& clusters, TMatchedTracks const& primTracks, TMatchedSecondaries const& secTracks)
  {
    fRegistry.fill(HIST("Cluster/before/hNgamma"), clusters.size(), collision.weight());
    int ngAfter = 0;
    for (const auto& cluster : clusters) {
      // Fill the cluster properties before applying any cuts
      if (!fEMCCut.IsSelectedEMCal(EMCPhotonCut::EMCPhotonCuts::kDefinition, cluster)) {
        continue;
      }
      auto primTracksPerCluster = primTracks.sliceBy(perEMCClusterMT, cluster.globalIndex());
      auto secTracksPerCluster = secTracks.sliceBy(perEMCClusterMS, cluster.globalIndex());
      o2::aod::pwgem::photonmeson::utils::clusterhistogram::fillClusterHistograms<0>(&fRegistry, cluster, cfgDo2DQA, collision.weight(), primTracksPerCluster, secTracksPerCluster);

      // Apply cuts one by one and fill in hClusterQualityCuts histogram
      fRegistry.fill(HIST("Cluster/hClusterQualityCuts"), 0., cluster.e(), collision.weight());

      // Define two boleans to see, whether the cluster "survives" the EMC cluster cuts to later check, whether the cuts in this task align with the ones in EMCPhotonCut.h:
      bool survivesIsSelectedEMCalCuts = true; // Survives "manual" cuts listed in this task
      bool survivesIsSelectedCuts = true;

      survivesIsSelectedCuts = fEMCCut.IsSelected(cluster, primTracksPerCluster, secTracksPerCluster);
      for (int icut = 0; icut < static_cast<int>(EMCPhotonCut::EMCPhotonCuts::kNCuts); icut++) { // Loop through different cut observables
        auto specificcut = static_cast<EMCPhotonCut::EMCPhotonCuts>(icut);
        if (specificcut == EMCPhotonCut::EMCPhotonCuts::kTM || specificcut == EMCPhotonCut::EMCPhotonCuts::kSecondaryTM) {
          // will do track matching cuts extra later or never depending on chosen process function
          continue;
        }
        if (!fEMCCut.IsSelectedEMCal(specificcut, cluster)) { // Check whether cluster passes this cluster requirement, if not, fill why in the next row
          fRegistry.fill(HIST("Cluster/hClusterQualityCuts"), icut + 1, cluster.e(), collision.weight());
          survivesIsSelectedEMCalCuts = false;
        }
      }
      if (primTracksPerCluster.size() > 0) {
        const auto closestPrimTrack = primTracksPerCluster.begin();
        fRegistry.fill(HIST("Cluster/hDeltaEtaPhiClosestPrimTracks"), closestPrimTrack.deltaEta(), closestPrimTrack.deltaPhi(), cluster.e());
        fRegistry.fill(HIST("Cluster/hDeltaEtaPhiClosestPrimTracksTrackPt"), closestPrimTrack.deltaEta(), closestPrimTrack.deltaPhi(), closestPrimTrack.trackPt());
      }
      for (const auto& matchedPrimTrack : primTracksPerCluster) {
        fRegistry.fill(HIST("Cluster/hDeltaEtaPhiAllPrimTracks"), matchedPrimTrack.deltaEta(), matchedPrimTrack.deltaPhi(), cluster.e());
        fRegistry.fill(HIST("Cluster/hDeltaEtaPhiAllPrimTracksTrackPt"), matchedPrimTrack.deltaEta(), matchedPrimTrack.deltaPhi(), matchedPrimTrack.trackPt());
      }
      if (secTracksPerCluster.size() > 0) {
        const auto closestSecTrack = secTracksPerCluster.begin();
        fRegistry.fill(HIST("Cluster/hDeltaEtaPhiClosestSecTracks"), closestSecTrack.deltaEta(), closestSecTrack.deltaPhi(), cluster.e());
        fRegistry.fill(HIST("Cluster/hDeltaEtaPhiClosestSecTracksTrackPt"), closestSecTrack.deltaEta(), closestSecTrack.deltaPhi(), closestSecTrack.trackPt());
      }
      for (const auto& matchedSecTrack : secTracksPerCluster) {
        fRegistry.fill(HIST("Cluster/hDeltaEtaPhiAllSecTracks"), matchedSecTrack.deltaEta(), matchedSecTrack.deltaPhi(), cluster.e());
        fRegistry.fill(HIST("Cluster/hDeltaEtaPhiAllSecTracksTrackPt"), matchedSecTrack.deltaEta(), matchedSecTrack.deltaPhi(), matchedSecTrack.trackPt());
      }

      if (!fEMCCut.IsSelectedEMCal(EMCPhotonCut::EMCPhotonCuts::kTM, cluster, primTracksPerCluster)) { // Check whether cluster passes this cluster requirement, if not, fill why in the next row
        fRegistry.fill(HIST("Cluster/hClusterQualityCuts"), static_cast<double>(EMCPhotonCut::EMCPhotonCuts::kTM) + 1, cluster.e(), collision.weight());
        survivesIsSelectedEMCalCuts = false;
      }
      if (!fEMCCut.IsSelectedEMCal(EMCPhotonCut::EMCPhotonCuts::kSecondaryTM, cluster, secTracksPerCluster)) { // Check whether cluster passes this cluster requirement, if not, fill why in the next row
        fRegistry.fill(HIST("Cluster/hClusterQualityCuts"), static_cast<double>(EMCPhotonCut::EMCPhotonCuts::kSecondaryTM) + 1, cluster.e(), collision.weight());
        survivesIsSelectedEMCalCuts = false;
      }

      if (survivesIsSelectedCuts != survivesIsSelectedEMCalCuts) [[unlikely]] {
        LOGF(info, "Cummulative application of IsSelectedEMCal cuts does not equal the IsSelected result");
        LOGF(info, "Cummulative application of IsSelectedEMCal resulted in %d", survivesIsSelectedEMCalCuts);
        LOGF(info, "IsSelected resulted in %d", survivesIsSelectedCuts);
      }

      if (survivesIsSelectedCuts) {
        o2::aod::pwgem::photonmeson::utils::clusterhistogram::fillClusterHistograms<1>(&fRegistry, cluster, cfgDo2DQA, collision.weight(), primTracksPerCluster, secTracksPerCluster);
        fRegistry.fill(HIST("Cluster/hClusterQualityCuts"), static_cast<double>(EMCPhotonCut::EMCPhotonCuts::kNCuts) + 1., cluster.e(), collision.weight());
        ++ngAfter;
      }
    }
    fRegistry.fill(HIST("Cluster/after/hNgamma"), ngAfter, collision.weight());
  }

  void processQC(MyCollisions const& collisions, EMCalPhotons const& clusters)
  {
    for (const auto& collision : collisions) {

      if (!isEventGood(collision)) {
        continue;
      }

      auto clustersPerColl = clusters.sliceBy(perCollisionEMC, collision.collisionId());
      doClusterQA(collision, clustersPerColl);

    } // end of collision loop
  } // end of process

  void processQCTM(MyCollisions const& collisions, EMCalPhotons const& clusters, MinMTracks const& matchedPrims, MinMSTracks const& matchedSeconds)
  {
    for (const auto& collision : collisions) {

      if (!isEventGood(collision)) {
        continue;
      }

      auto clustersPerColl = clusters.sliceBy(perCollisionEMC, collision.collisionId());
      doClusterQAWithTM(collision, clustersPerColl, matchedPrims, matchedSeconds);

    } // end of collision loop
  } // end of process

  void processDummy(MyCollisions const&) {}

  PROCESS_SWITCH(EmcalQC, processQC, "run EMCal QC", false);
  PROCESS_SWITCH(EmcalQC, processQCTM, "run EMCal QC with track matching", false);
  PROCESS_SWITCH(EmcalQC, processDummy, "Dummy function", true);
};

WorkflowSpec defineDataProcessing(ConfigContext const& context)
{
  return WorkflowSpec{adaptAnalysisTask<EmcalQC>(context)};
}
