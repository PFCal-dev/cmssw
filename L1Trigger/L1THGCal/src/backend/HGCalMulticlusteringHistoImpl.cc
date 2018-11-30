#include "L1Trigger/L1THGCal/interface/backend/HGCalMulticlusteringHistoImpl.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalShowerShape.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "TFile.h"
#include "TH2D.h"

HGCalMulticlusteringHistoImpl::HGCalMulticlusteringHistoImpl( const edm::ParameterSet& conf ) :
    dr_(conf.getParameter<double>("dR_multicluster")),
    ptC3dThreshold_(conf.getParameter<double>("minPt_multicluster")),
    multiclusterAlgoType_(conf.getParameter<string>("type_multicluster")),    
    nBinsRHisto_(conf.getParameter<unsigned>("nBins_R_histo_multicluster")),
    nBinsPhiHisto_(conf.getParameter<unsigned>("nBins_Phi_histo_multicluster")),
    binsSumsHisto_(conf.getParameter< std::vector<unsigned> >("binSumsHisto")),
    histoThreshold_(conf.getParameter<double>("threshold_histo_multicluster")),
    neighbour_weights_(conf.getParameter< std::vector<double> >("neighbour_weights"))
{    
  
    if(multiclusterAlgoType_=="HistoMaxC3d"){
      multiclusteringAlgoType_ = HistoMaxC3d;
    }else if(multiclusterAlgoType_=="HistoModifiedMaxC3d"){
      multiclusteringAlgoType_ = HistoModifiedMaxC3d;
    }else if(multiclusterAlgoType_=="HistoThresholdC3d"){
      multiclusteringAlgoType_ = HistoThresholdC3d;
    }else if(multiclusterAlgoType_=="HistoInterpolatedMaxC3d"){
      multiclusteringAlgoType_ = HistoInterpolatedMaxC3d;
    }else {
      throw cms::Exception("HGCTriggerParameterError")
	<< "Unknown Multiclustering type '" << multiclusterAlgoType_;
    } 

    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster dR: " << dr_;  
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster minimum transverse-momentum: " << ptC3dThreshold_;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster number of R-bins for the histo algorithm: " << nBinsRHisto_<<endl;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster number of Phi-bins for the histo algorithm: " << nBinsPhiHisto_<<endl;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster MIPT threshold for histo threshold algorithm: " << histoThreshold_<<endl;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster type of multiclustering algortihm: " << multiclusterAlgoType_;
    id_.reset( HGCalTriggerClusterIdentificationFactory::get()->create("HGCalTriggerClusterIdentificationBDT") );
    id_->initialize(conf.getParameter<edm::ParameterSet>("EGIdentification"));
    if(multiclusterAlgoType_.find("Histo")!=std::string::npos && nBinsRHisto_!=binsSumsHisto_.size()) throw cms::Exception("Inconsistent nBins_R_histo_multicluster and binSumsHisto size in HGCalMulticlustering");
    if(neighbour_weights_.size()!=neighbour_weights_size_) throw cms::Exception("Inconsistent size of neighbour weights vector in HGCalMulticlustering");


}



float HGCalMulticlusteringHistoImpl::dR( const l1t::HGCalCluster & clu,
					 const GlobalPoint & seed) const
{

    Basic3DVector<float> seed_3dv( seed );
    GlobalPoint seed_proj( seed_3dv / seed.z() );
    return (seed_proj - clu.centreProj() ).mag();

}





HGCalMulticlusteringHistoImpl::Histogram HGCalMulticlusteringHistoImpl::fillHistoClusters( const std::vector<edm::Ptr<l1t::HGCalCluster>> & clustersPtrs ){


    Histogram histoClusters; //key[0] = z.side(), key[1] = bin_R, key[2] = bin_phi

    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){

                histoClusters[{{z_side, bin_R, bin_phi}}] = 0;

            }

        }

    }


    for(auto & clu : clustersPtrs){

        float ROverZ = sqrt( pow(clu->centreProj().x(),2) + pow(clu->centreProj().y(),2) );
        int bin_R = int( (ROverZ-kROverZMin_) * nBinsRHisto_ / (kROverZMax_-kROverZMin_) );
        int bin_phi = int( (reco::reduceRange(clu->phi())+M_PI) * nBinsPhiHisto_ / (2*M_PI) );

        histoClusters[{{clu->zside(), bin_R, bin_phi}}]+=clu->mipPt();

    }

    return histoClusters;

}




HGCalMulticlusteringHistoImpl::Histogram HGCalMulticlusteringHistoImpl::fillSmoothPhiHistoClusters( const Histogram & histoClusters,
								     const vector<unsigned> & binSums ){

    Histogram histoSumPhiClusters; //key[0] = z.side(), key[1] = bin_R, key[2] = bin_phi

    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            int nBinsSide = (binSums[bin_R]-1)/2;
            float R1 = kROverZMin_ + bin_R*(kROverZMax_-kROverZMin_);
            float R2 = R1 + (kROverZMax_-kROverZMin_);
            double area = 0.5 * (pow(R2,2)-pow(R1,2)) * (1+0.5*(1-pow(0.5,nBinsSide))); // Takes into account different area of bins in different R-rings + sum of quadratic weights used

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){

                float content = histoClusters.at({{z_side,bin_R,bin_phi}});

                for(int bin_phi2=1; bin_phi2<=nBinsSide; bin_phi2++ ){

                    int binToSumLeft = bin_phi - bin_phi2;
                    if( binToSumLeft<0 ) binToSumLeft += nBinsPhiHisto_;
                    int binToSumRight = bin_phi + bin_phi2;
                    if( binToSumRight>=int(nBinsPhiHisto_) ) binToSumRight -= nBinsPhiHisto_;

                    content += histoClusters.at({{z_side,bin_R,binToSumLeft}}) / pow(2,bin_phi2); // quadratic kernel
                    content += histoClusters.at({{z_side,bin_R,binToSumRight}}) / pow(2,bin_phi2); // quadratic kernel

                }

                histoSumPhiClusters[{{z_side,bin_R,bin_phi}}] = content/area;

            }

        }

    }

    return histoSumPhiClusters;

}






HGCalMulticlusteringHistoImpl::Histogram HGCalMulticlusteringHistoImpl::fillSmoothRPhiHistoClusters( const Histogram & histoClusters ){

    Histogram histoSumRPhiClusters; //key[0] = z.side(), key[1] = bin_R, key[2] = bin_phi

    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            float weight = (bin_R==0 || bin_R==int(nBinsRHisto_)-1) ? 1.5 : 2.; //Take into account edges with only one side up or down

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){

                float content = histoClusters.at({{z_side,bin_R,bin_phi}});
                float contentDown = bin_R>0 ? histoClusters.at({{z_side,bin_R-1,bin_phi}}) : 0 ;
                float contentUp = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,bin_phi}}) : 0;

                histoSumRPhiClusters[{{z_side,bin_R,bin_phi}}] = (content + 0.5*contentDown + 0.5*contentUp)/weight;

            }

        }

    }

    return histoSumRPhiClusters;

}




std::vector<std::pair<GlobalPoint, double > > HGCalMulticlusteringHistoImpl::computeModifiedMaxSeeds( const Histogram & histoClusters ){

  std::vector<std::pair<GlobalPoint, double > > seedPositions;

    std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, bool> > > primarySeedPositions;
    std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, bool> > > secondarySeedPositions;
    std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, bool> > > vetoPositions;

    //Search for primary seeds
    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){

                float MIPT_seed = histoClusters.at({{z_side,bin_R,bin_phi}});
                bool isMax = MIPT_seed > histoThreshold_;

                float MIPT_S = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,bin_phi}}) : 0;
                float MIPT_N = bin_R>0 ? histoClusters.at({{z_side,bin_R-1,bin_phi}}) : 0;

                int binLeft = bin_phi - 1;
                if( binLeft<0 ) binLeft += nBinsPhiHisto_;
                int binRight = bin_phi + 1;
                if( binRight>=int(nBinsPhiHisto_) ) binRight -= nBinsPhiHisto_;

                float MIPT_W = histoClusters.at({{z_side,bin_R,binLeft}});
                float MIPT_E = histoClusters.at({{z_side,bin_R,binRight}});
                float MIPT_NW = bin_R>0 ? histoClusters.at({{z_side,bin_R-1,binLeft}}) : 0;
                float MIPT_NE = bin_R>0 ?histoClusters.at({{z_side,bin_R-1,binRight}}) : 0;
                float MIPT_SW = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,binLeft}}) : 0;
                float MIPT_SE = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,binRight}}) : 0;

                isMax &= MIPT_seed>=MIPT_S;
                isMax &= MIPT_seed>MIPT_N;
                isMax &= MIPT_seed>=MIPT_E;
                isMax &= MIPT_seed>=MIPT_SE;
                isMax &= MIPT_seed>=MIPT_NE;
                isMax &= MIPT_seed>MIPT_W;
                isMax &= MIPT_seed>MIPT_SW;
                isMax &= MIPT_seed>MIPT_NW;

                if(isMax){
                    float ROverZ_seed = kROverZMin_ + (bin_R+0.5) * (kROverZMax_-kROverZMin_)/nBinsRHisto_;
                    float phi_seed = -M_PI + (bin_phi+0.5) * 2*M_PI/nBinsPhiHisto_;
                    float x_seed = ROverZ_seed*cos(phi_seed);
                    float y_seed = ROverZ_seed*sin(phi_seed);

                    seedPositions.emplace_back( std::make_pair( GlobalPoint(x_seed,y_seed,z_side), MIPT_seed) );
		    primarySeedPositions[bin_R][bin_phi][z_side] =  true;


		    vetoPositions[bin_R][binLeft][z_side] = true;
		    vetoPositions[bin_R][binRight][z_side] = true;
		    if ( bin_R>0 ) {
		      vetoPositions[bin_R-1][bin_phi][z_side] = true;
		      vetoPositions[bin_R-1][binRight][z_side] = true;
		      vetoPositions[bin_R-1][binLeft][z_side] = true;
		    }
		    if ( bin_R<(int(nBinsRHisto_)-1) ) {
		      vetoPositions[bin_R+1][bin_phi][z_side] = true;
		      vetoPositions[bin_R+1][binRight][z_side] = true;
		      vetoPositions[bin_R+1][binLeft][z_side] = true;
		    }

                }
		// else{
		//     primarySeedPositions[bin_R, bin_phi,z_side] = false;
		// }

            }

        }

    }





    //Search for secondary seeds

    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){

	      //Cannot be a secondary seed if already a primary seed, or adjacent to primary seed
	      if ( primarySeedPositions[bin_R][bin_phi][z_side] || vetoPositions[bin_R][bin_phi][z_side] ) continue;

                float MIPT_seed = histoClusters.at({{z_side,bin_R,bin_phi}});
                bool isMax = MIPT_seed > histoThreshold_;

                float MIPT_S = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,bin_phi}}) : 0;
                float MIPT_N = bin_R>0 ? histoClusters.at({{z_side,bin_R-1,bin_phi}}) : 0;

                int binLeft = bin_phi - 1;
                if( binLeft<0 ) binLeft += nBinsPhiHisto_;
                int binRight = bin_phi + 1;
                if( binRight>=int(nBinsPhiHisto_) ) binRight -= nBinsPhiHisto_;

                float MIPT_W = histoClusters.at({{z_side,bin_R,binLeft}});
                float MIPT_E = histoClusters.at({{z_side,bin_R,binRight}});
                float MIPT_NW = bin_R>0 ? histoClusters.at({{z_side,bin_R-1,binLeft}}) : 0;
                float MIPT_NE = bin_R>0 ?histoClusters.at({{z_side,bin_R-1,binRight}}) : 0;
                float MIPT_SW = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,binLeft}}) : 0;
                float MIPT_SE = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,binRight}}) : 0;


		if (  !vetoPositions[bin_R+1][bin_phi][z_side]  ) isMax &= MIPT_seed>=MIPT_S;
		if (  !vetoPositions[bin_R-1][bin_phi][z_side]  ) isMax &= MIPT_seed>MIPT_N;
		if (  !vetoPositions[bin_R][binRight][z_side]  ) isMax &= MIPT_seed>=MIPT_E;
		if (  !vetoPositions[bin_R+1][binRight][z_side]  ) isMax &= MIPT_seed>=MIPT_SE;
		if (  !vetoPositions[bin_R-1][binRight][z_side]  ) isMax &= MIPT_seed>=MIPT_NE;
		if (  !vetoPositions[bin_R][binLeft][z_side]  ) isMax &= MIPT_seed>MIPT_W;
		if (  !vetoPositions[bin_R+1][binLeft][z_side]  ) isMax &= MIPT_seed>MIPT_SW;
		if (  !vetoPositions[bin_R-1][binLeft][z_side]  ) isMax &= MIPT_seed>MIPT_NW;

                if(isMax){
                    float ROverZ_seed = kROverZMin_ + (bin_R+0.5) * (kROverZMax_-kROverZMin_)/nBinsRHisto_;
                    float phi_seed = -M_PI + (bin_phi+0.5) * 2*M_PI/nBinsPhiHisto_;
                    float x_seed = ROverZ_seed*cos(phi_seed);
                    float y_seed = ROverZ_seed*sin(phi_seed);
                    seedPositions.emplace_back( std::make_pair( GlobalPoint(x_seed,y_seed,z_side), MIPT_seed) );
		    secondarySeedPositions[bin_R][bin_phi][z_side] =  true;
                }

            }

        }

    }





    //TESTING

    // TFile * file = new TFile ("ModMax.root", "RECREATE");
    // TH2D * hist2D = new TH2D( "ModMax","",50,-0.5,49.5,250,-0.5,249.5);

    // //    for(int z_side : {-1,1}){
    // for(int z_side : {1}){
    //     for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){
    //         for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){

    //             float MIPT_seed = histoClusters.at({{z_side,bin_R,bin_phi}});
    // 		//		std::cout << MIPT_seed << ", " << z_side <<", "<<bin_R<<", "<<bin_phi << ", " << primarySeedPositions[bin_R][bin_phi][z_side] << ", " << secondarySeedPositions[bin_R][bin_phi][z_side] << std::endl;
    // 		if ( primarySeedPositions[bin_R][bin_phi][z_side] ) hist2D->Fill ( bin_R, bin_phi, MIPT_seed );
    // 		if ( secondarySeedPositions[bin_R][bin_phi][z_side] ) hist2D->Fill ( bin_R, bin_phi, MIPT_seed );
    // 	    }
    // 	}
    // }

    // hist2D->Write();
    // file->Close();

    return seedPositions;




}




std::vector<std::pair<GlobalPoint, double > > HGCalMulticlusteringHistoImpl::computeMaxSeeds( const Histogram & histoClusters ){

    // TFile * file = new TFile ("DefaultMax.root", "RECREATE");
    // TH2D * hist2D = new TH2D( "DefaultMax","",50,-0.5,49.5,250,-0.5,249.5);


    std::vector<std::pair<GlobalPoint, double > > seedPositions;

    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){

                float MIPT_seed = histoClusters.at({{z_side,bin_R,bin_phi}});
                bool isMax = MIPT_seed>0;

                float MIPT_S = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,bin_phi}}) : 0;
                float MIPT_N = bin_R>0 ? histoClusters.at({{z_side,bin_R-1,bin_phi}}) : 0;

                int binLeft = bin_phi - 1;
                if( binLeft<0 ) binLeft += nBinsPhiHisto_;
                int binRight = bin_phi + 1;
                if( binRight>=int(nBinsPhiHisto_) ) binRight -= nBinsPhiHisto_;

                float MIPT_W = histoClusters.at({{z_side,bin_R,binLeft}});
                float MIPT_E = histoClusters.at({{z_side,bin_R,binRight}});
                float MIPT_NW = bin_R>0 ? histoClusters.at({{z_side,bin_R-1,binLeft}}) : 0;
                float MIPT_NE = bin_R>0 ?histoClusters.at({{z_side,bin_R-1,binRight}}) : 0;
                float MIPT_SW = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,binLeft}}) : 0;
                float MIPT_SE = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,binRight}}) : 0;

                isMax &= MIPT_seed>=MIPT_S;
                isMax &= MIPT_seed>MIPT_N;
                isMax &= MIPT_seed>=MIPT_E;
                isMax &= MIPT_seed>=MIPT_SE;
                isMax &= MIPT_seed>=MIPT_NE;
                isMax &= MIPT_seed>MIPT_W;
                isMax &= MIPT_seed>MIPT_SW;
                isMax &= MIPT_seed>MIPT_NW;



                if(isMax){


                    float ROverZ_seed = kROverZMin_ + (bin_R+0.5) * (kROverZMax_-kROverZMin_)/nBinsRHisto_;
                    float phi_seed = -M_PI + (bin_phi+0.5) * 2*M_PI/nBinsPhiHisto_;
                    float x_seed = ROverZ_seed*cos(phi_seed);
                    float y_seed = ROverZ_seed*sin(phi_seed);
                    seedPositions.emplace_back( std::make_pair( GlobalPoint(x_seed,y_seed,z_side), MIPT_seed) );
		    // if ( z_side == 1){
		    //   hist2D->Fill ( bin_R, bin_phi, MIPT_seed );
		    // }

		}
            }

        }

    }
    //    hist2D->Write();
    //    file->Close();

    return seedPositions;

}


std::vector<std::pair<GlobalPoint, double > > HGCalMulticlusteringHistoImpl::computeInterpolatedMaxSeeds( const Histogram & histoClusters ){
  

  std::vector<std::pair<GlobalPoint, double > > seedPositions;
  
    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){
              
                float MIPT_seed = histoClusters.at({{z_side,bin_R,bin_phi}});
                bool isMax = MIPT_seed > histoThreshold_;		
                
                float MIPT_S = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,bin_phi}}) : 0;
                float MIPT_N = bin_R>0 ? histoClusters.at({{z_side,bin_R-1,bin_phi}}) : 0;
                
                int binLeft = bin_phi - 1;
                if( binLeft<0 ) binLeft += nBinsPhiHisto_;
                int binRight = bin_phi + 1;
                if( binRight>=int(nBinsPhiHisto_) ) binRight -= nBinsPhiHisto_;
                
                float MIPT_W = histoClusters.at({{z_side,bin_R,binLeft}});
                float MIPT_E = histoClusters.at({{z_side,bin_R,binRight}});
                
                float MIPT_NW = bin_R>0 ? histoClusters.at({{z_side,bin_R-1,binLeft}}) : 0;
                float MIPT_NE = bin_R>0 ? histoClusters.at({{z_side,bin_R-1,binRight}}) : 0;
                float MIPT_SW = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,binLeft}}) : 0;
                float MIPT_SE = bin_R<(int(nBinsRHisto_)-1) ? histoClusters.at({{z_side,bin_R+1,binRight}}) : 0;
                
                float MIPT_pred = neighbour_weights_.at(0) * MIPT_NW + neighbour_weights_.at(1) * MIPT_N + neighbour_weights_.at(2) * MIPT_NE
                  + neighbour_weights_.at(3) * MIPT_W + neighbour_weights_.at(5) * MIPT_E + neighbour_weights_.at(6) * MIPT_SW
                  + neighbour_weights_.at(7) * MIPT_S + neighbour_weights_.at(8) * MIPT_SE;
                
                isMax &= MIPT_seed>=MIPT_pred;
                
                if(isMax){
                  float ROverZ_seed = kROverZMin_ + (bin_R+0.5) * (kROverZMax_-kROverZMin_)/nBinsRHisto_;
                  float phi_seed = -M_PI + (bin_phi+0.5) * 2*M_PI/nBinsPhiHisto_;
                  float x_seed = ROverZ_seed*cos(phi_seed);
                  float y_seed = ROverZ_seed*sin(phi_seed);
		  seedPositions.emplace_back( std::make_pair( GlobalPoint(x_seed,y_seed,z_side), MIPT_seed) );
                }
                
            }
            
        }
        
    }
    
    return seedPositions;
    
}


std::vector<std::pair<GlobalPoint, double > > HGCalMulticlusteringHistoImpl::computeThresholdSeeds( const Histogram & histoClusters ){


    std::vector<std::pair<GlobalPoint, double > > seedPositions;

    for(int z_side : {-1,1}){

        for(int bin_R = 0; bin_R<int(nBinsRHisto_); bin_R++){

            for(int bin_phi = 0; bin_phi<int(nBinsPhiHisto_); bin_phi++){

                float MIPT_seed = histoClusters.at({{z_side,bin_R,bin_phi}});

                bool isSeed = MIPT_seed > histoThreshold_;

                if(isSeed){
                    float ROverZ_seed = kROverZMin_ + (bin_R+0.5) * (kROverZMax_-kROverZMin_)/nBinsRHisto_;
                    float phi_seed = -M_PI + (bin_phi+0.5) * 2*M_PI/nBinsPhiHisto_;
                    float x_seed = ROverZ_seed*cos(phi_seed);
                    float y_seed = ROverZ_seed*sin(phi_seed);
                    seedPositions.emplace_back( std::make_pair( GlobalPoint(x_seed,y_seed,z_side), MIPT_seed) );
                }

            }

        }

    }

    return seedPositions;

}



std::vector<l1t::HGCalMulticluster> HGCalMulticlusteringHistoImpl::clusterSeedMulticluster(const std::vector<edm::Ptr<l1t::HGCalCluster>> & clustersPtrs,
											   const std::vector<std::pair<GlobalPoint, double> > & seeds){

  // bool splitEnergyApproach= true;
  // bool distanceApproach= false;
  bool splitEnergyApproach= false;
  bool distanceApproach= true;

    std::map<int,l1t::HGCalMulticluster> mapSeedMulticluster;
    std::vector<l1t::HGCalMulticluster> multiclustersTmp;
    for(auto & clu : clustersPtrs){

        HGCalDetId cluDetId( clu->detId() );
        int z_side = cluDetId.zside();

        double minDist = dr_;
	std::vector<int> targetSeeds;
	std::vector<double> targetSeedsEnergy;
	//        int targetSeed = -1;
        for( unsigned int iseed=0; iseed<seeds.size(); iseed++ ){

            if( z_side*seeds[iseed].first.z()<0) continue;

            double d = this->dR(*clu, seeds[iseed].first);

	    if ( d < dr_ ){
	      if ( splitEnergyApproach ){
		targetSeeds.push_back( iseed );
		targetSeedsEnergy.push_back( seeds[iseed].second );
	      }
	      if ( distanceApproach ){
		if(d<minDist){
		  minDist = d;
		  //		  targetSeed = iseed;
		  if ( targetSeeds.size()==0 ) {
		    targetSeeds.push_back( iseed );
		    targetSeedsEnergy.push_back( seeds[iseed].second );
		  }
		  else {
		    targetSeeds.at(0) = iseed ;
		    targetSeedsEnergy.at(0) = ( seeds[iseed].second );
		  }
		}
	      }

	    }


        }

	//	if(targetSeed<0) continue;
	if(targetSeeds.size()==0) continue;
	//Loop over target seeds and divide up the clusters energy
	
	double totalTargetSeedEnergy = 0;
	for (auto energy: targetSeedsEnergy){
	  totalTargetSeedEnergy+=energy;
	}

	// if(mapSeedMulticluster[targetSeed].size()==0) mapSeedMulticluster[targetSeed] = l1t::HGCalMulticluster(clu);
	// else mapSeedMulticluster[targetSeed].addConstituent(clu);

	for (unsigned int seed = 0; seed < targetSeeds.size(); seed++){
	  
	  double seedWeight = 1;
       	  if ( splitEnergyApproach ) seedWeight = targetSeedsEnergy[seed]/totalTargetSeedEnergy;
	  //         std::cout << "seed weight = " << seedWeight << std::endl;//quite small 0.03 for energy approach

	  if( mapSeedMulticluster[ targetSeeds[seed]].size()==0) mapSeedMulticluster[targetSeeds[seed]] = l1t::HGCalMulticluster(clu, seedWeight) ;
	  mapSeedMulticluster[targetSeeds[seed]].addConstituent(clu, true, seedWeight);	  
	  
	}
	
    }
    
    for(auto mclu : mapSeedMulticluster) multiclustersTmp.emplace_back(mclu.second);
    //    std::cout << "mcl size = " << multiclustersTmp.size() << std::endl;
    return multiclustersTmp;

}




void HGCalMulticlusteringHistoImpl::clusterizeHisto( const std::vector<edm::Ptr<l1t::HGCalCluster>> & clustersPtrs,
						     l1t::HGCalMulticlusterBxCollection & multiclusters,
						     const HGCalTriggerGeometryBase & triggerGeometry)
{

    /* put clusters into an r/z x phi histogram */
    Histogram histoCluster = fillHistoClusters(clustersPtrs); //key[0] = z.side(), key[1] = bin_R, key[2] = bin_phi, content = MIPTs summed along depth

    /* smoothen along the phi direction + normalize each bin to same area */
    Histogram smoothPhiHistoCluster = fillSmoothPhiHistoClusters(histoCluster,binsSumsHisto_);

    /* smoothen along the r/z direction */
    Histogram smoothRPhiHistoCluster = fillSmoothRPhiHistoClusters(histoCluster);

    /* seeds determined with local maximum criteria */
    //    std::vector<GlobalPoint> seedPositions;
    std::vector<std::pair<GlobalPoint, double> > seedPositionsEnergy;
    if (multiclusteringAlgoType_ == HistoMaxC3d) seedPositionsEnergy = computeMaxSeeds(smoothRPhiHistoCluster);
    else if(multiclusteringAlgoType_ == HistoThresholdC3d) seedPositionsEnergy = computeThresholdSeeds(smoothRPhiHistoCluster);
    else if(multiclusteringAlgoType_ == HistoInterpolatedMaxC3d) seedPositionsEnergy = computeInterpolatedMaxSeeds(smoothRPhiHistoCluster);
    else if(multiclusteringAlgoType_ == HistoModifiedMaxC3d) seedPositionsEnergy = computeModifiedMaxSeeds(smoothRPhiHistoCluster);
    /* clusterize clusters around seeds */
    std::vector<l1t::HGCalMulticluster> multiclustersTmp = clusterSeedMulticluster(clustersPtrs,seedPositionsEnergy);
    
    /* making the collection of multiclusters */
    finalizeClusters(multiclustersTmp, multiclusters, triggerGeometry);

}







void
HGCalMulticlusteringHistoImpl::
finalizeClusters(std::vector<l1t::HGCalMulticluster>& multiclusters_in,
            l1t::HGCalMulticlusterBxCollection& multiclusters_out, 
            const HGCalTriggerGeometryBase& triggerGeometry) {
    for(auto& multicluster : multiclusters_in) {
        // compute the eta, phi from its barycenter
        // + pT as scalar sum of pT of constituents
        double sumPt=multicluster.sumPt();
	//        const std::unordered_map<uint32_t, edm::Ptr<l1t::HGCalCluster>>& clusters = multicluster.constituents();
        // for(const auto& id_cluster : clusters) {
	//   sumPt += id_cluster.second->pt();
	//   //	  std::cout << "id_cluster.second->pt() = " << id_cluster.second->pt() << std::endl;
	// }

	// if (std::isnan( multicluster.centre().eta() )) std::cout << "NAN ETA1!" << std::endl;
	// if (std::isnan( multicluster.centre().phi() )) std::cout << "NAN PHI1!" << std::endl;
	//	std::cout << "sum pt = " << sumPt << std::endl;
	math::PtEtaPhiMLorentzVector multiclusterP4(  sumPt,
	//	math::PtEtaPhiMLorentzVector multiclusterP4(  multicluster.pt(),
                multicluster.centre().eta(),
                multicluster.centre().phi(),
                0. );
        multicluster.setP4( multiclusterP4 );

	// if ( std::isnan(multicluster.pt() )) std::cout << "NAN PT2!" << std::endl;
	// if ( std::isnan(multicluster.eta() )) std::cout << "NAN ETA2!" << std::endl;
	// if ( std::isnan(multicluster.phi() )) std::cout << "NAN PHI2!" << std::endl;

        if( multicluster.pt() > ptC3dThreshold_ ){
            //compute shower shapes
            multicluster.showerLength(shape_.showerLength(multicluster));
            multicluster.coreShowerLength(shape_.coreShowerLength(multicluster, triggerGeometry));
            multicluster.firstLayer(shape_.firstLayer(multicluster));
            multicluster.maxLayer(shape_.maxLayer(multicluster));
            multicluster.sigmaEtaEtaTot(shape_.sigmaEtaEtaTot(multicluster));
            multicluster.sigmaEtaEtaMax(shape_.sigmaEtaEtaMax(multicluster));
            multicluster.sigmaPhiPhiTot(shape_.sigmaPhiPhiTot(multicluster));
            multicluster.sigmaPhiPhiMax(shape_.sigmaPhiPhiMax(multicluster));
            multicluster.sigmaZZ(shape_.sigmaZZ(multicluster));
            multicluster.sigmaRRTot(shape_.sigmaRRTot(multicluster));
            multicluster.sigmaRRMax(shape_.sigmaRRMax(multicluster));
            multicluster.sigmaRRMean(shape_.sigmaRRMean(multicluster));
            multicluster.eMax(shape_.eMax(multicluster));
            // fill quality flag
            multicluster.setHwQual(id_->decision(multicluster));

	    //	    std::cout << "mc pt = " << multicluster.pt() << std::endl;

            multiclusters_out.push_back( 0, multicluster);
        }
    }
}
