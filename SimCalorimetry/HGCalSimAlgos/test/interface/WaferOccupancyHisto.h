#ifndef waferoccupancyhisto_h
#define waferoccupancyhisto_h

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH1F.h"
#include "TString.h"
#include "TMath.h"

#include <iostream>
#include <algorithm>

/**
   @class WaferOccupancyHisto
   @short book keeps a set of occupancy histograms for a wafer
*/
class WaferOccupancyHisto
{
public:
  
  typedef std::pair<int,int> UVKey_t;
  
  /**
     @short CTOR
  */
  WaferOccupancyHisto(int subdet, int layer,int u,int v,int ncells,edm::Service<TFileService> *fs) : ncells_(ncells), nEvents_(0)
    { 
      addWaferEquivalent(u,v);

      TString id(Form("sd%d_lay%d_%d_%d",subdet,layer,u,v));
      TFileDirectory subDir = (*fs)->mkdir(id.Data());
      adcH_ = subDir.make<TH1F>(id+"_adc",";q [MIP eq.];",100,0,5);
      adcH_->Sumw2();
      countH_ = subDir.make<TH1F>(id+"_counts",";Counts above threshold;",ncells,0,ncells);
      countH_->Sumw2();
      maxCountH_ = subDir.make<TH1F>(id+"_maxcounts",";Counts above threshold;",ncells,0,ncells);
      maxCountH_->Sumw2();
      maxNeighborCountH_.resize(6);
      for(size_t i=0; i<6; i++){
        maxNeighborCountH_[i] = subDir.make<TH1F>(id+Form("_maxcounts%d",int(i)),";Counts above threshold;",ncells,0,ncells);
        maxNeighborCountH_[i]->Sumw2();
      }
    }
  
  /**
     @short adds another wafer equivalent which will be integrated by this class
   */
  inline void addWaferEquivalent(int u, int v) 
  { 
    UVKey_t key(u,v);
    countMap_[key]=0;
  }

  /**
     @short accumulate counts for a new hit
  */
  inline void count(int u, int v,float adc,float thr=0.5,float fudgeFactor=0.8)
  {
    adcH_->Fill(adc);
    UVKey_t key(u,v);
    countMap_[key]+= (adc>=thr*fudgeFactor);
  }

  /**
     @short to be called once all hits have been counted
  */
  inline void analyze()
  {
    nEvents_++;
    
    int maxCounts(0);
    hotWaferKey_=UVKey_t(0,0);
    for(std::map<UVKey_t,int>::iterator it = countMap_.begin();
        it != countMap_.end();
        it++) {
      
      countH_->Fill( it->second );
      int nBelowThr(ncells_-it->second);
      countH_->Fill(0.,nBelowThr);     
      adcH_->Fill(0.,nBelowThr);

      if(it->second>maxCounts){
        maxCounts=it->second;
        hotWaferKey_=it->first;
      }      
    }

    maxCountH_->Fill(maxCounts);
    int nBelowThr(ncells_-maxCounts);
    maxCountH_->Fill(0.,nBelowThr);
  }

  /**
     @short set all to 0
   */
  inline void resetCounters()
  {
    for(std::map<UVKey_t,int>::iterator it = countMap_.begin();
        it != countMap_.end();
        it++) {
      countMap_[it->first]=0;
    }
    countHotSpotVec_.clear();
  }

  /**
     @short counts wafer data in nearest neighbors of the hotest wafer
   */
  inline void countHotWaferNeighbor(const std::map<UVKey_t,int> &otherCountMap)
  {
    for(std::map<UVKey_t,int>::const_iterator it = otherCountMap.begin();
        it != otherCountMap.end();
        it++) {

      if(it->first==hotWaferKey_) continue;
      int deltau(it->first.first-hotWaferKey_.first);
      int deltav(it->first.second-hotWaferKey_.second);
      bool isNeighbor( (deltau==1 && deltav==1) || (deltau==0 && deltav==1) || (deltau==-1 && deltav==0)
                       || (deltau==-1 && deltav==-1) || (deltau==0 && deltav==-1) || (deltau==1 && deltav==0) );
      if(!isNeighbor) continue;

      countHotSpotVec_.push_back( it->second );
    }
  }

  /**
     @short fills the hot spot neighbor counting histograms
   */
  inline void fillHotSpotNeighborCounts() 
  {
    std::sort(countHotSpotVec_.begin(),countHotSpotVec_.end(),std::greater<int>());
    for(size_t i=0; i<maxNeighborCountH_.size(); i++){
      float counts(countHotSpotVec_.size()>i ? countHotSpotVec_[i] : -1.);
      maxNeighborCountH_[i]->Fill(counts);
    }
  }
  

  /**
     @short normalize according to the number of events analyzed and number of equivalent wafers
     the factor of two is added as there are two endcaps
   */
  inline void endJob() 
  {
    if(nEvents_==0) return;
    int nWaferEq(weight());
    if(nWaferEq==0) return;

    //scale by the number of wafer equivalent
    float norm(2*nEvents_*nWaferEq);
    adcH_->Scale(1./norm);
    countH_->Scale(1./norm);

    //scale only by the number of events
    norm=float(2*nEvents_);
    maxCountH_->Scale(1./norm);
    for(auto h : maxNeighborCountH_) h->Scale(1./norm);
  }

  /**
     @short number of wafer equivalents
   */
  inline size_t weight() 
  {
    return countMap_.size();
  }

  /**
     @short returns a reference to the count map
   */
  inline const std::map<UVKey_t,int> &getCountMap() { return countMap_; }

  /**
     @short DTOR
   */
  ~WaferOccupancyHisto() {}

 
 private:
  int ncells_;
  TH1F *adcH_,*countH_,*maxCountH_;
  std::vector<TH1F *> maxNeighborCountH_;

  int nEvents_;
  std::map<UVKey_t,int> countMap_;
  UVKey_t hotWaferKey_;
  std::vector<int> countHotSpotVec_;
};

#endif
