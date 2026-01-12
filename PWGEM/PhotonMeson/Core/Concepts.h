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

/// \file Concepts.h
/// \brief Header with concepts for all tasks and helpers in the photon meson code.
/// \author M. Hemmer, marvin.hemmer@cern.ch

#ifndef PWGEM_PHOTONMESON_CORE_CONCEPTS_H_
#define PWGEM_PHOTONMESON_CORE_CONCEPTS_H_

#include <Framework/ASoA.h>

#include <concepts>
#include <vector>

// general concept
template <typename T>
concept HasEMEvent = requires(T t) {
  t.emevent();
};

template <typename T>
concept HasMomentum3D = requires(T t) {
  { t.px() } -> std::convertible_to<float>;
  { t.py() } -> std::convertible_to<float>;
  { t.pz() } -> std::convertible_to<float>;
};

template <typename T>
concept HasPtEtaPhi = requires(T t) {
  { t.pt() } -> std::convertible_to<float>;
  { t.eta() } -> std::convertible_to<float>;
  { t.phi() } -> std::convertible_to<float>;
};

template <typename T>
concept HasEnergy = requires(T t) {
  { t.e() } -> std::convertible_to<float>;
};

// Track likes
template <typename T>
concept IsTrackLike = requires(T t) {
  { t.collisionId() } -> std::integral;
  { t.sign() } -> std::integral;
  { t.dcaXY() } -> std::convertible_to<float>;
  { t.dcaZ() } -> std::convertible_to<float>;
};

template <typename T>
concept HasTPCPID = requires(T t) {
  { t.tpcNSigmaEl() } -> std::convertible_to<float>;
  { t.tpcNSigmaPi() } -> std::convertible_to<float>;
};

// V0 leg
template <typename T>
concept IsV0Leg = IsTrackLike<T> &&
                  HasMomentum3D<T> &&
                  requires(T t) {
                    { t.trackId() } -> std::integral;
                  };

// V0
template <typename T>
concept IsV0Photon = requires(T t) {
  { t.vx() } -> std::convertible_to<float>;
  { t.vy() } -> std::convertible_to<float>;
  { t.vz() } -> std::convertible_to<float>;
  { t.mGamma() } -> std::convertible_to<float>;
  { t.cospa() } -> std::convertible_to<float>;
} && HasPtEtaPhi<T>;

// calo concepts
template <typename T>
concept IsCaloCluster = requires(T t) {
  { t.e() } -> std::convertible_to<float>;
  { t.eta() } -> std::convertible_to<float>;
  { t.phi() } -> std::convertible_to<float>;
};

template <typename T>
concept HasShowerShape = requires(T t) {
  { t.m02() } -> std::convertible_to<float>;
  { t.nCells() } -> std::integral;
};

template <typename T>
concept IsEMCalCluster = IsCaloCluster<T> &&
                         HasShowerShape<T> &&
                         requires(T t) {
                           { t.isExotic() } -> std::same_as<bool>;
                         };

// Dielectron
template <typename T>
concept IsDielectron = requires(T t) {
  { t.mass() } -> std::convertible_to<float>;
  { t.opangle() } -> std::convertible_to<float>;
  { t.phiV() } -> std::convertible_to<float>;
} && HasPtEtaPhi<T>;

template <typename T>
concept IsTrackIterator = o2::soa::is_iterator<T> && requires(T t) {
  // Check that the *elements* of the container have the required methods:
  { t.deltaEta() } -> std::same_as<float>;
  { t.deltaPhi() } -> std::same_as<float>;
  { t.trackPt() } -> std::same_as<float>;
  { t.trackP() } -> std::same_as<float>;
};

template <typename T>
concept IsTrackContainer = o2::soa::is_table<T> && requires(T t) {
  // Check that the *elements* of the container have the required methods:
  { t.begin().deltaEta() } -> std::same_as<float>;
  { t.begin().deltaPhi() } -> std::same_as<float>;
  { t.begin().trackPt() } -> std::same_as<float>;
  { t.begin().trackP() } -> std::same_as<float>;
};

template <typename Cluster>
concept HasTrackMatching = requires(Cluster cluster) {
  // requires that the following are valid calls:
  { cluster.deltaEta() } -> std::convertible_to<std::vector<float>>;
  { cluster.deltaPhi() } -> std::convertible_to<std::vector<float>>;
  { cluster.trackpt() } -> std::convertible_to<std::vector<float>>;
  { cluster.trackp() } -> std::convertible_to<std::vector<float>>;
};

template <typename Cluster>
concept HasSecondaryMatching = requires(Cluster cluster) {
  // requires that the following are valid calls:
  { cluster.deltaEtaSec() } -> std::convertible_to<std::vector<float>>;
  { cluster.deltaPhiSec() } -> std::convertible_to<std::vector<float>>;
  { cluster.trackptSec() } -> std::convertible_to<std::vector<float>>;
  { cluster.trackpSec() } -> std::convertible_to<std::vector<float>>;
};

#endif // PWGEM_PHOTONMESON_CORE_CONCEPTS_H_
