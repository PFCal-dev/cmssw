#ifndef SimCalorimetry_HGCSimProducers_hgcdigitizerbase
#define SimCalorimetry_HGCSimProducers_hgcdigitizerbase

#include <array>
#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerTypes.h"

#include "SimCalorimetry/HGCalSimProducers/interface/HGCFEElectronics.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include <curand.h>

namespace hgc = hgc_digi;

namespace hgc_digi_utils {
  using hgc::HGCCellInfo;

  inline void addCellMetadata(HGCCellInfo& info,
		       const HcalGeometry* geom,
		       const DetId& detid ) {
    //base time samples for each DetId, initialized to 0
    info.size = 1.0;
    info.thickness = 1.0;
  }

  inline void addCellMetadata(HGCCellInfo& info,
		       const HGCalGeometry* geom,
		       const DetId& detid ) {
    const auto& dddConst = geom->topology().dddConstants();
    bool isHalf = (((dddConst.geomMode() == HGCalGeometryMode::Hexagon) ||
		    (dddConst.geomMode() == HGCalGeometryMode::HexagonFull)) ?
		   dddConst.isHalfCell(HGCalDetId(detid).wafer(),HGCalDetId(detid).cell()) :
		   false);
    //base time samples for each DetId, initialized to 0
    info.size = (isHalf ? 0.5 : 1.0);
    info.thickness = dddConst.waferType(detid);
  }

  inline void addCellMetadata(HGCCellInfo& info,
		       const CaloSubdetectorGeometry* geom,
		       const DetId& detid ) {
    if( DetId::Hcal == detid.det() ) {
      const HcalGeometry* hc = static_cast<const HcalGeometry*>(geom);
      addCellMetadata(info,hc,detid);
    } else {
      const HGCalGeometry* hg = static_cast<const HGCalGeometry*>(geom);
      addCellMetadata(info,hg,detid);
    }
  }

}


template <class DFr>
class HGCDigitizerBase {
 public:

  typedef DFr DigiType;

  typedef edm::SortedCollection<DFr> DColl;

  /**
     @short CTOR
  */
  HGCDigitizerBase(const edm::ParameterSet &ps);
 /**
    @short steer digitization mode
 */
  void run(std::unique_ptr<DColl> &digiColl, hgc::HGCSimHitDataAccumulator &simData,
	   const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
	   uint32_t digitizationType,CLHEP::HepRandomEngine* engine);

  /**
     @short getters
  */
  float keV2fC() const { return keV2fC_; }
  bool toaModeByEnergy() const { return (myFEelectronics_->toaMode()==HGCFEElectronics<DFr>::WEIGHTEDBYE); }
  float tdcOnset() const { return myFEelectronics_->getTDCOnset(); }
  std::array<float,3> tdcForToAOnset() const { return myFEelectronics_->getTDCForToAOnset(); }

  /**
     @short a trivial digitization: sum energies and digitize without noise
   */
  void runSimple(std::unique_ptr<DColl> &coll, hgc::HGCSimHitDataAccumulator &simData,
		 const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
		 CLHEP::HepRandomEngine* engine);

 /**
    @short a trivial digitization: sum energies and digitize without noise
  */
 void runSimpleOnGPU(std::unique_ptr<DColl> &coll, hgc::HGCSimHitDataAccumulator &simData,
		 const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds);

  /**
     @short prepares the output according to the number of time samples to produce
  */
  void updateOutput(std::unique_ptr<DColl> &coll, const DFr& rawDataFrame);


  /**
	     @shot prepares the output based on a NdetId x Nbunches array of words (GPU output)
	   */
	void updateOutput(const std::unordered_set<DetId>& validIds, const uint32_t *bxWord, std::unique_ptr<HGCDigitizerBase::DColl> &coll);


  /**
     @short to be specialized by top class
  */
  virtual void runDigitizer(std::unique_ptr<DColl> &coll, hgc::HGCSimHitDataAccumulator &simData,
			    const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
			    uint32_t digitizerType, CLHEP::HepRandomEngine* engine)
  {
    throw cms::Exception("HGCDigitizerBaseException") << " Failed to find specialization of runDigitizer";
  }

  /**
     @short DTOR
  */
  virtual ~HGCDigitizerBase()
    {
      if(isCUDAInit) endCUDA();
    };



 protected:

  //baseline configuration
  edm::ParameterSet myCfg_;

  //1keV in fC
  float keV2fC_;

  //noise level
  std::vector<float> noise_fC_;

  //charge collection efficiency
  std::vector<double> cce_;

  //front-end electronics model
  std::unique_ptr<HGCFEElectronics<DFr> > myFEelectronics_;

  //bunch time
  double bxTime_;

  //ZS thresholds
  float adcThreshold_fC_;

  //salturation
  float adcSaturation_fC_;

  //adcNbits
  uint32_t adcNbits_;

  //if true will put both in time and out-of-time samples in the event
  bool doTimeSamples_;

  //CUDA specific
  bool isCUDAInit;
  float *h_toa, *h_charge, *d_toa, *d_charge,*d_rand;
  uint16_t *h_type, *d_type;
  uint32_t *h_rawData, *d_rawData;
  curandGenerator_t d_gen;

  void initCUDA(const uint32_t N);
  void endCUDA();

};

#endif
