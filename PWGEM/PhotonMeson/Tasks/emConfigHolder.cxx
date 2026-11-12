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

/// \file emConfigHolder.cxx
/// \brief Central holder of shared PWGEM/PhotonMeson configurable groups.
///        Has no inputs and no process function, so it is independent of the AO2D content (derived or not).
///        Other tasks inherit the configurables via o2::aod::pwgem::pm::config::readGroupFromDevice().
/// \author Marvin Hemmer (marvin.hemmer@cern.ch) - Goethe University Frankfurt

#include "PWGEM/PhotonMeson/Core/EMConfigModules.h"

#include <Framework/AnalysisTask.h>
#include <Framework/ConfigContext.h>
#include <Framework/ControlService.h>
#include <Framework/InitContext.h>
#include <Framework/ProcessingContext.h>
#include <Framework/runDataProcessing.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::pwgem::pm;

struct EmConfigHolder {
  config::EventCutConfigurables eventcuts;
  config::EMCCutConfigurables emccuts;

  void init(InitContext&) {}

  /// The device only exists to carry the configurables, which consumers read in their init()
  /// from RunningWorkflowInfo. Without inputs it is driven by the DPL clock forever, so stop
  /// right away to not block the end of the workflow.
  void run(ProcessingContext& pc)
  {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& context)
{
  auto spec = adaptAnalysisTask<EmConfigHolder>(context);
  // spec.inputs.emplace_back("tfn", "TFN", "TFNumber");
  return WorkflowSpec{spec};
}
