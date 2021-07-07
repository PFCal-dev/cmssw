#ifndef __L1Trigger_L1THGCal_HGCalSortingTruncationConfig_SA_h__
#define __L1Trigger_L1THGCal_HGCalSortingTruncationConfig_SA_h__

#include <vector>

namespace l1thgcfirmware {

  class sortingTruncationAlgoConfig {
  public:
    sortingTruncationAlgoConfig(){};

    void setParameters() {};

    void setParameters(const sortingTruncationAlgoConfig& newConfig) {
      setParameters();
    }

    std::vector<double> truncation_byRoverZbin() const { return truncation_byRoverZbin_; }

  private:
    std::vector<double> truncation_byRoverZbin_;
  };

}  // namespace l1thgcfirmware

#endif
