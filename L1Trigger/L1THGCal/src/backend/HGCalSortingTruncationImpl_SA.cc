#include "L1Trigger/L1THGCal/interface/backend/HGCalSortingTruncationImpl_SA.h"

#include <cmath>
#include <algorithm>

HGCalSortingTruncationImplSA::HGCalSortingTruncationImplSA() {}

// void HGCalSortingTruncationImplSA::runAlgorithm() const {

// }

void HGCalSortingTruncationImplSA::sortAndTruncate_SA(
    const std::vector<l1thgcfirmware::HGCalMulticluster> inputMulticlusters,
    std::vector<l1thgcfirmware::HGCalMulticluster>& outputMulticlusters,
    const l1thgcfirmware::sortingTruncationAlgoConfig& configuration) const { 

  for (const auto& multicluster : inputMulticlusters) {
    outputMulticlusters.push_back(multicluster);
  }
  
  std::sort(outputMulticlusters.begin(), outputMulticlusters.end(), [](l1thgcfirmware::HGCalMulticluster & one, l1thgcfirmware::HGCalMulticluster & two){return one.sumPt() < two.sumPt();});
  
  

 }
