#ifndef RECOEGAMMA_EGAMMAISOLATIONALGOS_ELETKISOLFROMCANDS_H
#define RECOEGAMMA_EGAMMAISOLATIONALGOS_ELETKISOLFROMCANDS_H

#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//author S. Harper (RAL)
//this class does a simple calculation of the track isolation for a track with eta,
//phi and z vtx (typically the GsfTrack of the electron). It uses
//PFPackedCandidates as proxies for the tracks. It now has been upgraded to
//take general tracks as input also
//
//Note: due to rounding and space saving issues, the isolation calculated on tracks and
//PFPackedCandidates will differ slightly even if the same cuts are used. These differences
//are thought to be inconsquencial but as such its important to remake the PFPackedCandidates
//in RECO/AOD if you want to be consistant with miniAOD
//
//The track version is more for variables that are stored in the electron rather than
//objects which are recomputed on the fly for AOD/miniAOD
//
//Note, the tracks in miniAOD have additional cuts on and are missing algo information so this
//has to be taken into account when using them

//new for 9X:
//  1) its now possible to use generalTracks as input
//  2) adapted to 9X PF packed candidate improvements
//  2a) the PF packed candidates now store the pt of the track in addition to the pt of the
//      candidate. This means there is no need to pass in the electron collection any more
//      to try and undo the electron e/p combination when the candidate is an electron
//  2b) we now not only store the GsfTrack of the electron but also the general track of the electron
//      there are three input collections now:
//         packedPFCandidates : all PF candidates (with the track for ele candidates being the gsftrack)
//         lostTracks : all tracks which were not made into a PFCandidate but passed some preselection
//         lostTracks::eleTracks : KF electron tracks
//      as such to avoid double counting the GSF and KF versions of an electron track, we now need to
//      specify if the electron PF candidates are to be rejected in the sum over that collection or not.
//      Note in all this, I'm not concerned about the electron in questions track, that will be rejected,
//      I'm concerned about near by fake electrons which have been recoed by PF
//      This is handled by the PIDVeto, which obviously is only used/required when using PFCandidates

class EleTkIsolFromCands {
public:
  explicit EleTkIsolFromCands(const edm::ParameterSet& para);
  EleTkIsolFromCands(const EleTkIsolFromCands&) = default;
  ~EleTkIsolFromCands() = default;
  EleTkIsolFromCands& operator=(const EleTkIsolFromCands&) = default;

  static edm::ParameterSetDescription pSetDescript();

  enum class PIDVeto {
    NONE = 0,
    ELES,
    NONELES,
  };

  static PIDVeto pidVetoFromStr(const std::string& vetoStr);

  class SimpleTrack {
  public:
    SimpleTrack(reco::TrackBase const& trk) : pt{trk.pt()}, eta{trk.eta()}, phi{trk.phi()}, vz{trk.vz()} {}
    const double pt;
    const double eta;
    const double phi;
    const double vz;
  };

  // For each electron, we want to try out which tracks are in a cone around
  // it. However, this will get expensive if there are many electrons and
  // tracks (Phase II conditions). In particular, calling
  // reco::TrackBase::eta() many times is costy because eta is not precomputed.
  // To solve this, we first cache the tracks in a simpler data structure in
  // which eta is already computed (SimpleTrack). Furthermore, the tracks are
  // preselected by the cuts that can already be applied without considering
  // the electron. Note that this has to be done twice, because the required
  // preselection is different for barrel and endcap electrons.

  struct PreselectedTracks {
    const std::vector<SimpleTrack> withBarrelCuts;
    const std::vector<SimpleTrack> withEndcapCuts;
  };

  PreselectedTracks preselectTracks(reco::TrackCollection const& tracks) const;
  PreselectedTracks preselectTracks(pat::PackedCandidateCollection const& cands, PIDVeto = PIDVeto::NONE) const;

  struct Output {
    const int nTracks;
    const double ptSum;
  };

  Output operator()(const reco::TrackBase& electronTrack, const PreselectedTracks& tracks) const;

private:
  struct TrkCuts {
    float minPt;
    float minDR2;
    float maxDR2;
    float minDEta;
    float maxDZ;
    float minHits;
    float minPixelHits;
    float maxDPtPt;
    std::vector<reco::TrackBase::TrackQuality> allowedQualities;
    std::vector<reco::TrackBase::TrackAlgorithm> algosToReject;
    explicit TrkCuts(const edm::ParameterSet& para);
    static edm::ParameterSetDescription pSetDescript();
  };

  static bool passPIDVeto(const int pdgId, const EleTkIsolFromCands::PIDVeto pidVeto);

  static std::vector<SimpleTrack> preselectTracksWithCuts(reco::TrackCollection const& tracks, TrkCuts const& cuts);
  static std::vector<SimpleTrack> preselectTracksWithCuts(pat::PackedCandidateCollection const& cands,
                                                          TrkCuts const& cuts,
                                                          PIDVeto = PIDVeto::NONE);

  static bool passTrackPreselection(const reco::TrackBase& trk, const TrkCuts& cuts);
  static bool passMatchingToElectron(
      SimpleTrack const& trk, const TrkCuts& cuts, double eleEta, double elePhi, double eleVZ);
  //no qualities specified, accept all, ORed
  static bool passQual(const reco::TrackBase& trk, const std::vector<reco::TrackBase::TrackQuality>& quals);
  static bool passAlgo(const reco::TrackBase& trk, const std::vector<reco::TrackBase::TrackAlgorithm>& algosToRej);

  TrkCuts barrelCuts_, endcapCuts_;
};

#endif
