#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSiNoiseMap.h"

//
HGCalSiNoiseMap::HGCalSiNoiseMap() :
  encpScale_(840.),
  encScale_(1.60217646E-4)
{
  encsParam_[q80fC]  = {636.,  15.6, 0.0328};
  encsParam_[q160fC] = {1045., 8.74, 0.0685};
  encsParam_[q320fC] = {1915., 2.79, 0.0878};

  ileakParam_ = {0.993,-42.668};

  cellCapacitance_[HGCSiliconDetId::waferType::HGCalFine]=50;
  cellCapacitance_[HGCSiliconDetId::waferType::HGCalCoarseThin]=65;
  cellCapacitance_[HGCSiliconDetId::waferType::HGCalCoarseThick]=45;

  cellVolume_[HGCSiliconDetId::waferType::HGCalFine]=0.52*(120.e-4);
  cellVolume_[HGCSiliconDetId::waferType::HGCalCoarseThin]=1.18*(200.e-4);
  cellVolume_[HGCSiliconDetId::waferType::HGCalCoarseThick]=1.18*(300.e-4);

  cceParam_[HGCSiliconDetId::waferType::HGCalFine]={6e+15, -8.73841e-17, -2.27227e-17};
  cceParam_[HGCSiliconDetId::waferType::HGCalCoarseThin]={1.5e+15, -3.29653e-16, -8.17497e-17};
  cceParam_[HGCSiliconDetId::waferType::HGCalCoarseThick]={4e+14, -8.31403e-16, -4.97935e-16};
}

//
HGCalSiNoiseMap::SiCellOpCharacteristics HGCalSiNoiseMap::getSiCellOpCharacteristics(SignalRange_t srange,const HGCSiliconDetId &cellId,double &radius) {

  SiCellOpCharacteristics siop;

  //decode cell properties
  int layer(cellId.layer());
  if(cellId.subdet()==DetId::HGCalEE) layer=(layer-1)/2+1;
  HGCSiliconDetId::waferType cellThick( HGCSiliconDetId::waferType(cellId.type()) );
  double cellCap(cellCapacitance_[cellThick]);
  double cellVol(cellVolume_[cellThick]);

  //get fluence
  if(getDoseMap().empty()) return siop;
  std::array<double, 8> radii{ {radius,pow(radius,2),pow(radius,3),pow(radius,4),0.,0.,0.,0.} };
  siop.fluence=getFluenceValue(cellId.subdet(),layer,radii);
  siop.lnfluence=log(siop.fluence);

  //leakage current [muA]
  siop.ileak=exp(ileakParam_[0]*siop.lnfluence+ileakParam_[1])*cellVol*1e6;

  siop.cce=siop.fluence<cceParam_[cellThick][0] ? 1+cceParam_[cellThick][1]*siop.fluence : cceParam_[cellThick][2]*siop.fluence+(cceParam_[cellThick][1]-cceParam_[cellThick][2])*cceParam_[cellThick][0]+1;

  //build noise estimate
  double enc_p(encpScale_*sqrt(siop.ileak));
  double enc_s(encsParam_[srange][0]+encsParam_[srange][1]*cellCap+encsParam_[srange][2]*pow(cellCap,2));
  siop.noise=hypot(enc_p,enc_s)*encScale_;

  return siop;
}
