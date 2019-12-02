#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFELinearizationImpl.h"

HGCalVFELinearizationImpl::HGCalVFELinearizationImpl(const edm::ParameterSet& conf)
    : linLSB_(conf.getParameter<double>("linLSB")),
      adcsaturation_(conf.getParameter<double>("adcsaturation")),
      tdcnBits_(conf.getParameter<uint32_t>("tdcnBits")),
      tdcOnsetfC_(conf.getParameter<double>("tdcOnsetfC")),
      adcnBits_(conf.getParameter<uint32_t>("adcnBits")),
      tdcsaturation_(conf.getParameter<double>("tdcsaturation")),
      linnBits_(conf.getParameter<uint32_t>("linnBits")),
      oot_coefficients_(conf.getParameter<std::vector<double>>("oot_coefficients")) {
  const int kOot_order = 2;
  if (oot_coefficients_.size() != kOot_order) {
    throw cms::Exception("BadConfiguration") << "OOT subtraction needs " << kOot_order << " coefficients";
  }
  adcLSB_ = adcsaturation_ / pow(2., adcnBits_);
  tdcLSB_ = tdcsaturation_ / pow(2., tdcnBits_);
  linMax_ = (0x1 << linnBits_) - 1;
}

void HGCalVFELinearizationImpl::linearize(const std::vector<HGCDataFrame<DetId, HGCSample>>& dataframes,
                                          std::vector<std::pair<DetId, uint32_t>>& linearized_dataframes) {
  const int kIntimeSample = 2;

  for (const auto& frame : dataframes) {  //loop on DIGI
    double amplitude = 0.;
    uint32_t amplitude_int = 0;
    unsigned det = frame.id().det();
    if (det == DetId::Forward || det == DetId::HGCalEE || det == DetId::HGCalHSi) {
      if (frame[kIntimeSample].mode()) {  //TOT mode
        amplitude = (floor(tdcOnsetfC_ / adcLSB_) + 1.0) * adcLSB_ + double(frame[kIntimeSample].data()) * tdcLSB_;
      } else {  //ADC mode
        double data = frame[kIntimeSample].data();
        // applies OOT PU subtraction only in the ADC mode
        if (!frame[kIntimeSample - 1].mode()) {
          data += oot_coefficients_[kIntimeSample - 1] * frame[kIntimeSample - 1].data();
          if (!frame[kIntimeSample - 2].mode()) {
            data += oot_coefficients_[kIntimeSample - 2] * frame[kIntimeSample - 2].data();
          }
        }
        amplitude = std::max(0., data) * adcLSB_;
      }

      amplitude_int = uint32_t(floor(amplitude / linLSB_ + 0.5));
    } else if (det == DetId::Hcal || det == DetId::HGCalHSc) {
      // no linearization here. Take the raw ADC data
      amplitude_int = frame[kIntimeSample].data();
    }
    if (amplitude_int == 0)
      continue;
    if (amplitude_int > linMax_)
      amplitude_int = linMax_;

    linearized_dataframes.push_back(std::make_pair(frame.id(), amplitude_int));
  }
}
