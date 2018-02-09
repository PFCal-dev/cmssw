///
/// \class HGCalTowerMap3DImpl
///
/// \author: Thomas Strebler
///
/// Description: first iteration of HGCal Tower Maps

#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTowerMap3DImpl.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

constexpr double HGCalTowerMap3DImpl::kEtaMax_;
constexpr double HGCalTowerMap3DImpl::kEtaMin_;
constexpr double HGCalTowerMap3DImpl::kPhiMax_;
constexpr double HGCalTowerMap3DImpl::kPhiMin_;


HGCalTowerMap3DImpl::HGCalTowerMap3DImpl( const edm::ParameterSet& conf ) :
  nEtaBins_(conf.getParameter<unsigned>("nEtaBins")),
  nPhiBins_(conf.getParameter<unsigned>("nPhiBins")),
  etaBins_(conf.getParameter<std::vector<double> >("etaBins")),
  phiBins_(conf.getParameter<std::vector<double> >("phiBins"))
{
  edm::LogInfo("HGCalTowerMap3DImpl") << "Number of eta bins for the tower maps: " << nEtaBins_<<endl;
  edm::LogInfo("HGCalTowerMap3DImpl") << "Number of phi bins for the tower maps: " << nPhiBins_<<endl;

  if(etaBins_.size()>0 && etaBins_.size()!=nEtaBins_+1){
    edm::LogError("HGCalTowerMap3DImpl") << "nEtaBins for the tower mapsnot consistent with etaBins size"<<endl;
  }
  if(phiBins_.size()>0 && phiBins_.size()!=nPhiBins_+1){
    edm::LogError("HGCalTowerMap3DImpl") << "nPhiBins for the tower mapsnot consistent with phiBins size"<<endl;
  }
    
  //If no custom binning specified, assume uniform one

  if(etaBins_.size()==0){
    double eta_step = (kEtaMax_ - kEtaMin_)/nEtaBins_;
    for(unsigned int i=0; i<nEtaBins_; i++)
      etaBins_.emplace_back( kEtaMin_ + eta_step*i );
    etaBins_.emplace_back( kEtaMax_ );
  }

  if(phiBins_.size()==0){
    double phi_step = (kPhiMax_ - kPhiMin_)/nPhiBins_;
    for(unsigned int i=0; i<nPhiBins_; i++)
      phiBins_.emplace_back( kPhiMin_ + phi_step*i );
    phiBins_.emplace_back( kPhiMax_ );
  }

  edm::LogInfo("HGCalTowerMap3DImpl") << "Eta bins for the tower maps: {";
  for(auto eta : etaBins_) edm::LogInfo("HGCalTowerMap3DImpl") << eta << ",";
  edm::LogInfo("HGCalTowerMap3DImpl") << "}" <<endl;
  edm::LogInfo("HGCalTowerMap3DImpl") << "Phi bins for the tower maps: {";
  for(auto phi : phiBins_) edm::LogInfo("HGCalTowerMap3DImpl") << phi << ",";
  edm::LogInfo("HGCalTowerMap3DImpl") << "}" <<endl;  

}



void HGCalTowerMap3DImpl::buildTowerMapMulticlusters(const std::vector<edm::Ptr<l1t::HGCalCluster>> & clustersPtr,
					 l1t::HGCalMulticlusterBxCollection & towerMaps
				       ){

  map<pair<int,int>, l1t::HGCalMulticluster> MulticlusterTowerMap; //key = iEtaPhiBin

  for( std::vector<edm::Ptr<l1t::HGCalCluster>>::const_iterator cl = clustersPtr.begin(); cl != clustersPtr.end(); ++cl ){
    int iEtaBin = -nEtaBins_;
    int iPhiBin = -nPhiBins_;
    double absEta = fabs((*cl)->eta());
    double signEta = (*cl)->eta()>0 ? 1 : -1;
    double phi = normalizedPhi((*cl)->phi());

    if(absEta<kEtaMin_) absEta=kEtaMin_;
    else if(absEta>kEtaMax_) absEta=kEtaMax_;

    for(unsigned int i = 0; i<nEtaBins_; i++){
      if(absEta>etaBins_[i] && absEta<etaBins_[i+1]){
	iEtaBin = signEta*(i+1);
	break;
      }
    }
    
    for(unsigned int i = 0; i<phiBins_.size(); i++){
      if(phi>phiBins_[i] && phi<phiBins_[i+1]){
	iPhiBin = i;
	break;
      }
    }


    pair<int,int> iEtaPhiBin = make_pair(iEtaBin,iPhiBin);    
    if(MulticlusterTowerMap[iEtaPhiBin].constituents().size()==0) MulticlusterTowerMap[iEtaPhiBin] = l1t::HGCalMulticluster( *cl );
    else MulticlusterTowerMap[iEtaPhiBin].addConstituent( *cl );

  }


  /* store clusters in the persistent collection */
  towerMaps.resize(0, MulticlusterTowerMap.size());
  unsigned i = 0;
  for(auto multicluster : MulticlusterTowerMap){   
    towerMaps.set( 0, i, multicluster.second);
    i++;
  }


}



