#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1THGCal/interface/HGCalAlgoWrapperBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalSortingTruncationImpl_SA.h"

#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalMulticluster_SA.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalSortingTruncationWrapper : public HGCalSortingTruncationWrapperBase {
public:
  HGCalSortingTruncationWrapper(const edm::ParameterSet& conf);
  ~HGCalSortingTruncationWrapper() override {}

  void configure(const std::pair<const edm::EventSetup&, const edm::ParameterSet&>& configuration) override;

  void process(const l1t::HGCalMulticlusterBxCollection&, l1t::HGCalMulticlusterBxCollection&) const override;

private:
  void convertCMSSWInputs(const l1t::HGCalMulticlusterBxCollection& multiclusters,
                          l1thgcfirmware::HGCalMulticlusterSACollection& multiclusters_SA) const;
  void convertAlgorithmOutputs(const l1thgcfirmware::HGCalMulticlusterSACollection& multiclusters_out,
                               l1t::HGCalMulticlusterBxCollection& multiclustersBXCollection) const;

  unsigned packMulticlusterClusterId(const unsigned multicluster_id, const unsigned cluster_id ) const;

  void unpackMulticlusterClusterId(unsigned id, unsigned& multicluster_id, unsigned& cluster_id ) const;

  void sortAndTruncate(const l1thgcfirmware::HGCalMulticlusterSACollection& inputMulticlusters,
                       l1thgcfirmware::HGCalMulticlusterSACollection& outputMulticlusters) const;

  void eventSetup(const edm::EventSetup& es) { triggerTools_.eventSetup(es); }

  HGCalTriggerTools triggerTools_;
  HGCalSortingTruncationImplSA theAlgo_;
  l1thgcfirmware::sortingTruncationAlgoConfig theConfiguration_;

  unsigned kClusterIdMask = 0xFFFFF;
  unsigned kMulticlusterIdMask = 0xFFFFF;
  unsigned kClusterIdOffset = 0xFFF;
  unsigned kMulticlusterIdOffset = 20;


};

HGCalSortingTruncationWrapper::HGCalSortingTruncationWrapper(const edm::ParameterSet& conf)
  : HGCalSortingTruncationWrapperBase(conf),
    theAlgo_(), 
    theConfiguration_() {}


void HGCalSortingTruncationWrapper::convertCMSSWInputs(const l1t::HGCalMulticlusterBxCollection& multiclusters,
                                                       l1thgcfirmware::HGCalMulticlusterSACollection& multiclusters_SA) const {
  //std::vector<l1t::HGCalMulticluster> multiclusters;

  multiclusters_SA.reserve(multiclusters.size());

  for (unsigned int imulticluster = 0; imulticluster < multiclusters.size(); ++imulticluster) {
    bool firstConstituent = true;
    unsigned int clusterIndex = 0;
    for (const auto& constituent : multiclusters[imulticluster].constituents()) {
      if (firstConstituent) {
        multiclusters_SA.emplace_back(
            l1thgcfirmware::HGCalMulticluster(l1thgcfirmware::HGCalCluster(constituent.second->centreProj().x(),
                                                           constituent.second->centreProj().y(),
                                                           constituent.second->centreProj().z(),
                                                           triggerTools_.zside(constituent.second->detId()),
                                                           triggerTools_.layerWithOffset(constituent.second->detId()),
                                                           constituent.second->eta(),
                                                           constituent.second->phi(),
                                                           constituent.second->pt(),
                                                           constituent.second->mipPt(),
                                                           packMulticlusterClusterId(imulticluster, clusterIndex)),
                                      1.));

      } else {
        multiclusters_SA.at(imulticluster).addConstituent(l1thgcfirmware::HGCalCluster(constituent.second->centreProj().x(),
                                                           constituent.second->centreProj().y(),
                                                           constituent.second->centreProj().z(),
                                                           triggerTools_.zside(constituent.second->detId()),
                                                           triggerTools_.layerWithOffset(constituent.second->detId()),
                                                           constituent.second->eta(),
                                                           constituent.second->phi(),
                                                           constituent.second->pt(),
                                                           constituent.second->mipPt(),
                                                           packMulticlusterClusterId(imulticluster, clusterIndex)),
							  1.);

      }
      firstConstituent = false;
      ++clusterIndex;
    }
  }

}

void HGCalSortingTruncationWrapper::convertAlgorithmOutputs(
    const std::vector<l1thgcfirmware::HGCalMulticluster>& multiclusters_out,
    l1t::HGCalMulticlusterBxCollection& multiclustersBXCollection ) const {


  std::vector<l1t::HGCalMulticluster> multiclusters;
  multiclusters.reserve(multiclusters_out.size());
  // for (unsigned int imulticluster = 0; imulticluster < multiclusters_out.size(); ++imulticluster) {
  //   bool firstConstituent = true;
  //   for (const auto& constituent : multiclusters_out[imulticluster].constituents()) {
  //     if (firstConstituent) {
  //       multiclusters.emplace_back(l1t::HGCalMulticluster(clustersPtrs.at(constituent.index_cmssw()), 1.));
  //     } else {
  //       multiclusters.at(imulticluster).addConstituent(clustersPtrs.at(constituent.index_cmssw()), 1.);
  //     }
  //     firstConstituent = false;
  //   }
  // }

  for (const auto& multicluster : multiclusters) {
    multiclustersBXCollection.push_back(0, multicluster);
  }

}

unsigned HGCalSortingTruncationWrapper::packMulticlusterClusterId(const unsigned multicluster_id, const unsigned cluster_id ) const{

  unsigned packed_value = 0;

  packed_value |= ((multicluster_id & kMulticlusterIdMask) << kMulticlusterIdOffset);
  packed_value |= ((cluster_id & kClusterIdMask) << kClusterIdOffset);

  return packed_value;
  
}

  void HGCalSortingTruncationWrapper::unpackMulticlusterClusterId(unsigned id, unsigned& multicluster_id, unsigned& cluster_id ) const{

  multicluster_id = (id >> kMulticlusterIdOffset) & kMulticlusterIdMask;
  cluster_id = (id >> kClusterIdOffset) & kClusterIdMask;
  
}

void HGCalSortingTruncationWrapper::process(const l1t::HGCalMulticlusterBxCollection& inputMulticlusters,
                                            l1t::HGCalMulticlusterBxCollection& outputMulticlusters) const {
  l1thgcfirmware::HGCalMulticlusterSACollection multiclusters_SA;
  convertCMSSWInputs(inputMulticlusters, multiclusters_SA);

  l1thgcfirmware::HGCalMulticlusterSACollection multiclusters_finalized_SA;
  sortAndTruncate(multiclusters_SA, multiclusters_finalized_SA);

  convertAlgorithmOutputs(multiclusters_finalized_SA, outputMulticlusters);
}

void HGCalSortingTruncationWrapper::sortAndTruncate(const l1thgcfirmware::HGCalMulticlusterSACollection& inputMulticlusters,
                                                    l1thgcfirmware::HGCalMulticlusterSACollection& outputMulticlusters) const {
  // Call SA sorting
  theAlgo_.sortAndTruncate_SA(inputMulticlusters, outputMulticlusters, theConfiguration_);
}

void HGCalSortingTruncationWrapper::configure(
    const std::pair<const edm::EventSetup&, const edm::ParameterSet&>& configuration) {
  eventSetup(configuration.first);
  // theConfiguration_->setParameters( ... );
};

DEFINE_EDM_PLUGIN(HGCalSortingTruncationWrapperBaseFactory,
                  HGCalSortingTruncationWrapper,
                  "HGCalSortingTruncationWrapper");
