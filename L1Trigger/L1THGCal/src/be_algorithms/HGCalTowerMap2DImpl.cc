///
/// \class HGCalTowerMap2DImpl
///
/// \author: Thomas Strebler
///
/// Description: first iteration of HGCal Tower Maps

#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTowerMap2DImpl.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

constexpr double HGCalTowerMap2DImpl::kEtaMax_;
constexpr double HGCalTowerMap2DImpl::kEtaMin_;
constexpr double HGCalTowerMap2DImpl::kPhiMax_;
constexpr double HGCalTowerMap2DImpl::kPhiMin_;


HGCalTowerMap2DImpl::HGCalTowerMap2DImpl( const edm::ParameterSet& conf ) :
  nEtaBins_(conf.getParameter<unsigned>("nEtaBins")),
  nPhiBins_(conf.getParameter<unsigned>("nPhiBins")),
  etaBins_(conf.getParameter<std::vector<double> >("etaBins")),
  phiBins_(conf.getParameter<std::vector<double> >("phiBins")),
  layerWeights_(conf.getParameter< std::vector<double> >("layerWeights"))
{
  edm::LogInfo("HGCalTowerMap2DImpl") << "Number of eta bins for the tower maps: " << nEtaBins_<<endl;
  edm::LogInfo("HGCalTowerMap2DImpl") << "Number of phi bins for the tower maps: " << nPhiBins_<<endl;

  if(etaBins_.size()>0 && etaBins_.size()!=nEtaBins_+1){
    edm::LogError("HGCalTowerMap2DImpl") << "nEtaBins for the tower mapsnot consistent with etaBins size"<<endl;
  }
  if(phiBins_.size()>0 && phiBins_.size()!=nPhiBins_+1){
    edm::LogError("HGCalTowerMap2DImpl") << "nPhiBins for the tower mapsnot consistent with phiBins size"<<endl;
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

  edm::LogInfo("HGCalTowerMap2DImpl") << "Eta bins for the tower maps: {";
  for(auto eta : etaBins_) edm::LogInfo("HGCalTowerMap2DImpl") << eta << ",";
  edm::LogInfo("HGCalTowerMap2DImpl") << "}" <<endl;
  edm::LogInfo("HGCalTowerMap2DImpl") << "Phi bins for the tower maps: {";
  for(auto phi : phiBins_) edm::LogInfo("HGCalTowerMap2DImpl") << phi << ",";
  edm::LogInfo("HGCalTowerMap2DImpl") << "}" <<endl;  

}



void HGCalTowerMap2DImpl::buildTowerMapClusters(const std::vector<edm::Ptr<l1t::HGCalTriggerCell>> & triggerCellsPtrs,
					     l1t::HGCalClusterBxCollection & clusters
				       ){

  map<pair<pair<int,int>,int>, l1t::HGCalCluster> ClusterTowerMap; //key = pair(iEtaPhiBin,layer)	

  for( std::vector<edm::Ptr<l1t::HGCalTriggerCell>>::const_iterator tc = triggerCellsPtrs.begin(); tc != triggerCellsPtrs.end(); ++tc ){
    int iEtaBin = -nEtaBins_-1;
    int iPhiBin = -nPhiBins_-1;
    double absEta = fabs((*tc)->eta());
    double signEta = (*tc)->eta()>0 ? 1 : -1;
    double phi = normalizedPhi((*tc)->phi());

    if(absEta<kEtaMin_) absEta = kEtaMin_;
    else if(absEta>kEtaMax_) absEta = kEtaMax_;

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
    int layer = (*tc)->layer();
    if( (*tc)->subdetId()==HGCHEF ){
      layer += kLayersEE_;
    }
    else if( (*tc)->subdetId()==HGCHEB ){
      layer += kLayersFH_+kLayersEE_;
    }

    pair<pair<int,int>,int> key = make_pair(iEtaPhiBin,layer);
    if(ClusterTowerMap[key].constituents().size()==0) ClusterTowerMap[key] = l1t::HGCalCluster( *tc );
    else ClusterTowerMap[key].addConstituent( *tc );

  }

  /* calibrate and count non-zero clusters */
  unsigned i=0;
  for(auto cluster : ClusterTowerMap){
    calibratePt(cluster.second);    
    if(cluster.second.pt()>0) i++; //Layer calibration may set some layers to zero
  }

  /* store clusters in the persistent collection */
  clusters.resize(0, i);
  i=0;
  for(auto cluster : ClusterTowerMap){
    if(cluster.second.pt()>0){
      clusters.set( 0, i, cluster.second);
      i++;
    }
  }


}






void HGCalTowerMap2DImpl::calibratePt( l1t::HGCalCluster & cluster ){

    double calibPt=0.; 
    
    int layerN = -1;
    if( cluster.subdetId()==HGCEE ){
      layerN = cluster.layer();
    }
    else if( cluster.subdetId()==HGCHEF ){
      layerN = cluster.layer()+kLayersEE_;
    }
    else if( cluster.subdetId()==HGCHEB ){
      layerN = cluster.layer()+kLayersFH_+kLayersEE_;
    }

    calibPt = layerWeights_.at(layerN) * cluster.mipPt();

    math::PtEtaPhiMLorentzVector calibP4( calibPt,
                                          cluster.eta(),
                                          cluster.phi(),
                                          0. );

    cluster.setP4( calibP4 );

}
