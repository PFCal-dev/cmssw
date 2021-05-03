// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

// Geometry
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSciNoiseMap.h"

//ROOT headers
#include <TProfile2D.h>
#include <TH2F.h>
#include <TF1.h>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//STL headers
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>

#include "vdt/vdtMath.h"

//
// class declaration
//

class HGCHEbackSignalScalerAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit HGCHEbackSignalScalerAnalyzer(const edm::ParameterSet&);
  ~HGCHEbackSignalScalerAnalyzer() override;

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  void createBinning(const std::vector<DetId>&);
  void printBoundaries();

  // ----------member data ---------------------------
  edm::Service<TFileService> fs;

  std::string doseMap_;
  std::string sipmMap_;
  float pxFiringRate_;
  uint32_t nPEperMIP_;

  std::map<int, std::map<int, float>> layerRadiusMap_;
  std::map<int, double> layerMap_;
  std::map<int, std::vector<float>> hgcrocMap_;
  std::map<int, std::vector<int>> hgcrocNcellsMap_;

  const HGCalGeometry* gHGCal_;
  const HGCalDDDConstants* hgcCons_;

  int firstLayer_, lastLayer_;
  const float radiusMin_ = 70;   //cm
  const float radiusMax_ = 280;  //cm
  const int radiusBins_ = 25;  //nbins //cm
  const int nWedges_ = 72;
};

//
// constructors and destructor
//
HGCHEbackSignalScalerAnalyzer::HGCHEbackSignalScalerAnalyzer(const edm::ParameterSet& iConfig)
    : doseMap_(iConfig.getParameter<std::string>("doseMap")),
      sipmMap_(iConfig.getParameter<std::string>("sipmMap")),
      pxFiringRate_(iConfig.getParameter<double>("pxFiringRate")),
      nPEperMIP_(iConfig.getParameter<uint32_t>("nPEperMIP")) {
  usesResource("TFileService");
  fs->file().cd();
}

HGCHEbackSignalScalerAnalyzer::~HGCHEbackSignalScalerAnalyzer() {}

//
// member functions
//

// ------------ method called on each new Event  ------------
void HGCHEbackSignalScalerAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //get geometry
  edm::ESHandle<HGCalGeometry> geomhandle;
  iSetup.get<IdealGeometryRecord>().get("HGCalHEScintillatorSensitive", geomhandle);
  if (!geomhandle.isValid()) {
    edm::LogError("HGCHEbackSignalScalerAnalyzer") << "Cannot get valid HGCalGeometry Object";
    return;
  }
  gHGCal_ = geomhandle.product();
  const std::vector<DetId>& detIdVec = gHGCal_->getValidDetIds();
  std::cout << "total number of cells: " << detIdVec.size() << std::endl;

  //get ddd constants
  edm::ESHandle<HGCalDDDConstants> dddhandle;
  iSetup.get<IdealGeometryRecord>().get("HGCalHEScintillatorSensitive", dddhandle);
  if (!dddhandle.isValid()) {
    edm::LogError("HGCHEbackSignalScalerAnalyzer") << "Cannot initiate HGCalDDDConstants";
    return;
  }
  hgcCons_ = dddhandle.product();

  //setup maps
  createBinning(detIdVec);
  printBoundaries();
  //instantiate binning array
  std::vector<int> layvec;
  for (auto elem : layerMap_)
    layvec.push_back(elem.first);
  int minLay=*std::min_element(layvec.begin(), layvec.end());
  int maxLay=*std::max_element(layvec.begin(), layvec.end());
  size_t nLay(layvec.size());

  std::map<std::string,TH2F *> histos;

  histos["tilecount"] = fs->make<TH2F>("tilecount", ";Layer;Radius [cm];#tiles", nLay,minLay,maxLay+1, radiusBins_, radiusMin_, radiusMax_);
  histos["doseMap"] = fs->make<TH2F>("doseMap", ";Layer;Radius [cm];<D>", nLay,minLay,maxLay+1, radiusBins_, radiusMin_, radiusMax_);
  histos["fluenceMap"] =
      fs->make<TH2F>("fluenceMap", "f;Layer;Radius [cm];<f>", nLay,minLay,maxLay+1, radiusBins_, radiusMin_, radiusMax_);
  histos["scaleByDoseMap"] =
      fs->make<TH2F>("scaleByDoseMap", ";Layer;Radius [cm];<S>", nLay,minLay,maxLay+1, radiusBins_, radiusMin_, radiusMax_);
  histos["scaleByTileAreaMap"] = fs->make<TH2F>(
      "scaleByTileAreaMap", ";Layer;Radius [cm];<S>", nLay,minLay,maxLay+1, radiusBins_, radiusMin_, radiusMax_);
  histos["scaleByDoseAreaMap"] = fs->make<TH2F>(
      "scaleByDoseAreaMap", ";Layer;Radius [cm];<S>", nLay,minLay,maxLay+1, radiusBins_, radiusMin_, radiusMax_);
  histos["noiseByFluenceMap"] = fs->make<TH2F>(
      "noiseByFluenceMap", ";Layer;Radius [cm];<N>", nLay,minLay,maxLay+1, radiusBins_, radiusMin_, radiusMax_);
  histos["expNoiseMap"] = fs->make<TH2F>(
      "expNoiseMap", ";Layer;Radius [cm];<N>", nLay,minLay,maxLay+1, radiusBins_, radiusMin_, radiusMax_);
  histos["probNoiseAboveHalfMip"] = fs->make<TH2F>(
      "probNoiseAboveHalfMip", "probNoiseAboveHalfMip", nLay,minLay,maxLay+1, radiusBins_, radiusMin_, radiusMax_);

  histos["signalToNoiseFlatAreaMap"] = fs->make<TH2F>(
      "signalToNoiseFlatAreaMap", ";Layer;Radius [cm];<S/N>", nLay,minLay,maxLay+1, radiusBins_, radiusMin_, radiusMax_);
  histos["signalToNoiseDoseMap"] = fs->make<TH2F>(
      "signalToNoiseDoseMap", ";Layer;Radius [cm];<S/N>", nLay,minLay,maxLay+1, radiusBins_, radiusMin_, radiusMax_);
  histos["signalToNoiseAreaMap"] = fs->make<TH2F>(
      "signalToNoiseAreaMap", ";Layer;Radius [cm];<S/N>", nLay,minLay,maxLay+1, radiusBins_, radiusMin_, radiusMax_);
  histos["signalToNoiseDoseAreaMap"] = fs->make<TH2F>(
      "signalToNoiseDoseAreaMap", ";Layer;Radius [cm];<S/N>", nLay,minLay,maxLay+1, radiusBins_, radiusMin_, radiusMax_);
  histos["signalToNoiseDoseAreaSipmMap"] = fs->make<TH2F>("signalToNoiseDoseAreaSipmMap",
                                                          ";Layer;Radius [cm];<S/N>",
                                                          nLay,minLay,maxLay+1,
                                                          radiusBins_,
                                                          radiusMin_,
                                                          radiusMax_);

  histos["saturationMap"] =
      fs->make<TH2F>("saturationMap", ";Layer;Radius [cm];<Saturation>", nLay,minLay,maxLay+1, radiusBins_, radiusMin_, radiusMax_);

  for(auto h : histos) {
    h.second->Sumw2();
  }

  //book per layer plots
  std::map<int, TH1D*> probNoiseAboveHalfMip_layerMap;
  for (auto lay : layerRadiusMap_)
    probNoiseAboveHalfMip_layerMap[lay.first] = fs->make<TH1D>(Form("probNoiseAboveHalfMip_layer%d", lay.first),
                                                               "",
                                                               hgcrocMap_[lay.first].size() - 1,
                                                               hgcrocMap_[lay.first].data());

  //instantiate scaler
  HGCalSciNoiseMap scal;
  scal.setDoseMap(doseMap_, 0);
  scal.setSipmMap(sipmMap_);
  scal.setGeometry(gHGCal_);

  //loop over valid detId from the HGCHEback
  LogDebug("HGCHEbackSignalScalerAnalyzer") << "Total number of DetIDs: " << detIdVec.size();
  for (std::vector<DetId>::const_iterator myId = detIdVec.begin(); myId != detIdVec.end(); ++myId) {
    HGCScintillatorDetId scId(myId->rawId());

    int layer = scId.layer();

    GlobalPoint global = gHGCal_->getPosition(scId);
    double radius = std::sqrt(std::pow(global.x(), 2) + std::pow(global.y(), 2));

    double dose = scal.getDoseValue(DetId::HGCalHSc, layer, radius);
    double fluence = scal.getFluenceValue(DetId::HGCalHSc, layer, radius);

    auto dosePair = scal.scaleByDose(scId, radius,pxFiringRate_);

    float scaleFactorBySipmArea = scal.scaleBySipmArea(scId, radius);
    float scaleFactorByTileArea = scal.scaleByTileArea(scId, radius);
    float scaleFactorByDose = dosePair.first;
    float noiseByFluence = dosePair.second;
    float expNoise = noiseByFluence * sqrt(scaleFactorBySipmArea);

    TF1 mypois("mypois", "TMath::Poisson(x+[0],[0])", 0, 10000);  //subtract ped mean
    mypois.SetParameter(0, std::pow(expNoise, 2));
    double prob =
        mypois.Integral(nPEperMIP_ * scaleFactorByTileArea * scaleFactorBySipmArea * scaleFactorByDose * 0.5, 10000);

    int ilayer = scId.layer();
    int iradius = scId.iradiusAbs();

    std::pair<double, double> cellSize = hgcCons_->cellSizeTrap(scId.type(), scId.iradiusAbs());
    float rho = cellSize.first;

    histos["tilecount"]->Fill(ilayer,rho);
    histos["doseMap"]->Fill(ilayer, rho, dose);
    histos["fluenceMap"]->Fill(ilayer, rho, fluence);
    histos["scaleByDoseMap"]->Fill(ilayer, rho, scaleFactorByDose);
    histos["scaleByTileAreaMap"]->Fill(ilayer, rho, scaleFactorByTileArea);
    histos["scaleByDoseAreaMap"]->Fill(ilayer, rho, scaleFactorByDose * scaleFactorByTileArea);
    histos["noiseByFluenceMap"]->Fill(ilayer, rho, noiseByFluence);
    histos["expNoiseMap"]->Fill(ilayer, rho, expNoise);
    std::cout << ilayer << " " << rho << " " << dose << " " << fluence << " " 
              <<   nPEperMIP_ * scaleFactorByTileArea * scaleFactorByDose << " "
              << noiseByFluence << std::endl;

    histos["probNoiseAboveHalfMip"]->Fill(ilayer, rho, prob);
    histos["signalToNoiseFlatAreaMap"]->Fill(ilayer,rho,
                                            100 * scaleFactorByTileArea * scaleFactorBySipmArea / expNoise);
    histos["signalToNoiseDoseMap"]->Fill(ilayer, rho, 
                                         nPEperMIP_ * scaleFactorByDose / noiseByFluence);
    histos["signalToNoiseAreaMap"]->Fill(ilayer, rho, 
                                         nPEperMIP_ * scaleFactorByTileArea / noiseByFluence);
    histos["signalToNoiseDoseAreaMap"]->Fill(ilayer, rho,
                                             nPEperMIP_ * scaleFactorByTileArea * scaleFactorByDose / noiseByFluence);
    histos["signalToNoiseDoseAreaSipmMap"]->Fill(ilayer, rho,
                                                 nPEperMIP_ * scaleFactorByTileArea * scaleFactorByDose * scaleFactorBySipmArea / expNoise);
    histos["saturationMap"]->Fill(ilayer, rho,
                                  nPEperMIP_ * scaleFactorByTileArea * scaleFactorByDose * scaleFactorBySipmArea + std::pow(expNoise, 2));

    //fill per layer plots
    //float rpos = sqrt(global.x()*global.x() + global.y()*global.y());
    int rocbin = probNoiseAboveHalfMip_layerMap[ilayer]->FindBin(radius * 10);
    double scaleValue = prob / nWedges_ / hgcrocNcellsMap_[ilayer][rocbin - 1];
    probNoiseAboveHalfMip_layerMap[ilayer]->Fill(radius * 10, scaleValue);
  }

  //print boundaries for S/N < 5 --> define where sipms get more area
  std::cout << std::endl;
  std::cout << "S/N > 5 boundaries" << std::endl;
  std::cout << std::setw(5) << "layer" << std::setw(15) << "boundary" << std::endl;
  for (int xx = 1; xx < histos["signalToNoiseDoseAreaMap"]->GetNbinsX() + 1; ++xx) {
    bool print = true;
    float SoN = 0;
    for (int yy = 1; yy < histos["signalToNoiseDoseAreaMap"]->GetNbinsY() + 1; ++yy) {
      SoN = histos["signalToNoiseDoseAreaMap"]->GetBinContent(xx, yy);
      if (SoN > 5 && print == true) {
        std::cout << std::setprecision(5) << std::setw(5) << xx + 8 << std::setw(15)
                  << histos["signalToNoiseDoseAreaMap"]->GetYaxis()->GetBinLowEdge(yy) << std::endl;
        print = false;
      }
    }
  }

  //normalize histograms to reflect the average
  for(auto h : histos) {
    if(h.first!="tilecount")
      h.second->Divide( histos["tilecount"] );
  }
}

void HGCHEbackSignalScalerAnalyzer::createBinning(const std::vector<DetId>& detIdVec) {
  for (std::vector<DetId>::const_iterator myId = detIdVec.begin(); myId != detIdVec.end(); ++myId) {
    HGCScintillatorDetId scId(myId->rawId());

    int layer = std::abs(scId.layer());
    int radius = scId.iradiusAbs();
    GlobalPoint global = gHGCal_->getPosition(scId);

    //z-binning
    layerMap_[layer] = std::abs(global.z());

    //r-binning
    layerRadiusMap_[layer][radius] = (hgcCons_->cellSizeTrap(scId.type(), radius)).first;  //internal radius
  }
  //guess the last bins Z
  auto last = std::prev(layerMap_.end(), 1);
  auto lastbo = std::prev(layerMap_.end(), 2);
  layerMap_[last->first + 1] = last->second + (last->second - lastbo->second);

  //get external rad for the last bins r
  firstLayer_ = layerRadiusMap_.begin()->first;
  lastLayer_ = layerRadiusMap_.rbegin()->first;
  for (int lay = firstLayer_; lay <= lastLayer_; ++lay) {
    auto lastr = std::prev((layerRadiusMap_[lay]).end(), 1);
    layerRadiusMap_[lay][lastr->first + 1] =
        (hgcCons_->cellSizeTrap(hgcCons_->getTypeTrap(lay), lastr->first)).second;  //external radius
  }

  //implement by hand the approximate hgcroc boundaries
  std::vector<float> arr9 = {1537.0, 1790.7, 1997.1};
  hgcrocMap_[9] = arr9;
  std::vector<float> arr10 = {1537.0, 1790.7, 2086.2};
  hgcrocMap_[10] = arr10;
  std::vector<float> arr11 = {1537.0, 1790.7, 2132.2};
  hgcrocMap_[11] = arr11;
  std::vector<float> arr12 = {1537.0, 1790.7, 2179.2};
  hgcrocMap_[12] = arr12;
  std::vector<float> arr13 = {1378.2, 1503.9, 1790.7, 2132.2, 2326.6};
  hgcrocMap_[13] = arr13;
  std::vector<float> arr14 = {1378.2, 1503.9, 1790.7, 2132.2, 2430.4};
  hgcrocMap_[14] = arr14;
  std::vector<float> arr15 = {1183.0, 1503.9, 1790.7, 2132.2, 2538.8};
  hgcrocMap_[15] = arr15;
  std::vector<float> arr16 = {1183.0, 1503.9, 1790.7, 2132.2, 2594.8};
  hgcrocMap_[16] = arr16;
  std::vector<float> arr17 = {1183.0, 1503.9, 1790.7, 2132.2, 2594.8};
  hgcrocMap_[17] = arr17;
  std::vector<float> arr18 = {1183.0, 1503.9, 1790.7, 2132.2, 2594.8};
  hgcrocMap_[18] = arr18;
  std::vector<float> arr19 = {1037.8, 1157.5, 1503.9, 1790.7, 2132.2, 2594.8};
  hgcrocMap_[19] = arr19;
  std::vector<float> arr20 = {1037.8, 1157.5, 1503.9, 1790.7, 2132.2, 2594.8};
  hgcrocMap_[20] = arr20;
  std::vector<float> arr21 = {1037.8, 1157.5, 1503.9, 1790.7, 2132.2, 2594.8};
  hgcrocMap_[21] = arr21;
  std::vector<float> arr22 = {1037.8, 1157.5, 1503.9, 1790.7, 2132.2, 2484.0};
  hgcrocMap_[22] = arr22;
  std::vector<int> ncells9 = {64, 32};
  hgcrocNcellsMap_[9] = ncells9;
  std::vector<int> ncells10 = {64, 48};
  hgcrocNcellsMap_[10] = ncells10;
  std::vector<int> ncells11 = {64, 56};
  hgcrocNcellsMap_[11] = ncells11;
  std::vector<int> ncells12 = {64, 64};
  hgcrocNcellsMap_[12] = ncells12;
  std::vector<int> ncells13 = {40, 64, 64, 24};
  hgcrocNcellsMap_[13] = ncells13;
  std::vector<int> ncells14 = {40, 64, 64, 40};
  hgcrocNcellsMap_[14] = ncells14;
  std::vector<int> ncells15 = {88, 64, 64, 56};
  hgcrocNcellsMap_[15] = ncells15;
  std::vector<int> ncells16 = {88, 64, 64, 64};
  hgcrocNcellsMap_[16] = ncells16;
  std::vector<int> ncells17 = {88, 64, 64, 64};
  hgcrocNcellsMap_[17] = ncells17;
  std::vector<int> ncells18 = {88, 64, 64, 64};
  hgcrocNcellsMap_[18] = ncells18;
  std::vector<int> ncells19 = {40, 96, 64, 64, 64};
  hgcrocNcellsMap_[19] = ncells19;
  std::vector<int> ncells20 = {40, 96, 64, 64, 64};
  hgcrocNcellsMap_[20] = ncells20;
  std::vector<int> ncells21 = {40, 96, 64, 64, 64};
  hgcrocNcellsMap_[21] = ncells21;
  std::vector<int> ncells22 = {40, 96, 64, 64, 48};
  hgcrocNcellsMap_[22] = ncells22;
}

void HGCHEbackSignalScalerAnalyzer::printBoundaries() {
  std::cout << std::endl;
  std::cout << "z boundaries" << std::endl;
  std::cout << std::setw(5) << "layer" << std::setw(15) << "z-position" << std::endl;
  for (auto elem : layerMap_)
    std::cout << std::setprecision(5) << std::setw(5) << elem.first << std::setw(15) << elem.second << std::endl;

  std::cout << std::endl;
  std::cout << "r boundaries" << std::endl;
  std::cout << std::setw(5) << "layer" << std::setw(10) << "r-min" << std::setw(10) << "r-max" << std::endl;
  for (auto elem : layerRadiusMap_) {
    auto rMin = (elem.second).begin();
    auto rMax = (elem.second).rbegin();
    std::cout << std::setprecision(5) << std::setw(5) << elem.first << std::setw(10) << rMin->second << std::setw(10)
              << rMax->second << std::endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCHEbackSignalScalerAnalyzer);
