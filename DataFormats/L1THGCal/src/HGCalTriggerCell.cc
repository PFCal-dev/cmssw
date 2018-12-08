#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"

using namespace l1t;

HGCalTriggerCell::
HGCalTriggerCell( const LorentzVector& p4,
        int pt,
        int eta,
        int phi,
        int qual,
        uint32_t detid):
    L1Candidate(p4, pt, eta, phi, qual),
    detid_(detid)
{
}

HGCalTriggerCell::
HGCalTriggerCell( const PolarLorentzVector& p4,
        int pt,
        int eta,
        int phi,
        int qual,
        uint32_t detid):
    L1Candidate(p4, pt, eta, phi, qual),
    detid_(detid)
{
}

HGCalTriggerCell::
~HGCalTriggerCell() 
{
}

bool HGCalTriggerCell::operator==(const HGCalTriggerCell& rhs) const
{
  return L1Candidate::operator==(static_cast<const L1Candidate &>(rhs))
      && detid_.rawId() == rhs.detId()
      && position_ == rhs.position()
      && mipPt_ == rhs.mipPt()
      && uncompressedCharge_ == rhs.uncompressedCharge()
      && compressedCharge_ == rhs.compressedCharge();
}
