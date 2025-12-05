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
//
/// \file Pi0EtaToGammaGammaPCMDalitzEE.cxx
/// \brief This code loops over photons and makes pairs for neutral mesons analyses for PCM-Dalitz.
/// \author D. Sekihata, daiki.sekihata@cern.ch

#include "PWGEM/PhotonMeson/Core/Pi0EtaToGammaGamma.h"
#include "PWGEM/PhotonMeson/DataModel/gammaTables.h"
#include "PWGEM/PhotonMeson/Utils/PairUtilities.h"

#include <Framework/ASoA.h>
#include <Framework/ASoAHelpers.h>
#include <Framework/AnalysisTask.h>
#include <Framework/runDataProcessing.h>

using namespace o2;
using namespace o2::soa;
using namespace o2::aod;
using namespace o2::framework;
using namespace o2::aod::pwgem::photonmeson::photonpair;

using MyV0Photons = Filtered<Join<o2::aod::V0PhotonsKF, o2::aod::V0KFEMEventIds, o2::aod::V0PhotonsKFPrefilterBitDerived>>;
using MyPrimaryElectrons = Filtered<Join<o2::aod::EMPrimaryElectronsFromDalitz, o2::aod::EMPrimaryElectronEMEventIds, o2::aod::EMPrimaryElectronsPrefilterBitDerived>>;

template <>
void Pi0EtaToGammaGamma<PairType::kPCMDalitzEE, MyV0Photons, aod::V0Legs, MyPrimaryElectrons>::processAnalysis(Filtered<Join<EMEvents, EMEventsAlias, EMEventsMult, EMEventsCent, EMEventsQvec>> const& collisions, MyV0Photons const& v0photons, aod::V0Legs const& v0legs, MyPrimaryElectrons const& primaryElectrons)
{
  runPairing<PCMTag, DalitzEETag, CombinationsFullIndexPolicy>(collisions, v0photons, primaryElectrons, v0legs);
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<Pi0EtaToGammaGamma<PairType::kPCMDalitzEE, MyV0Photons, aod::V0Legs, MyPrimaryElectrons>>(cfgc, TaskName{"pi0eta-to-gammagamma-pcmdalitzee"}),
  };
}
