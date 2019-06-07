#ifndef waferoccupancyhisto_h
#define waferoccupancyhisto_h

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH1F.h"
#include "TString.h"
#include "TMath.h"

#include <exception>
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
      myUV_=UVKey_t(u,v);
      addWaferEquivalent(u,v);

      TString id(Form("sd%d_lay%d_%d_%d",subdet,layer,u,v));
      TFileDirectory subDir = (*fs)->mkdir(id.Data());
      adcH_ = subDir.make<TH1F>(id+"_adc",";q [MIP eq.];",100,0,5);
      adcH_->Sumw2();
      countH_ = subDir.make<TH1F>(id+"_counts",";Counts above threshold;",ncells,0,ncells);
      countH_->Sumw2();
      maxCountH_ = subDir.make<TH1F>(id+"_maxcounts",";Counts above threshold;",ncells,0,ncells);
      maxCountH_->Sumw2();
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
    if(adc>=thr*fudgeFactor)
      countMap_[key]=countMap_[key]+1;
  }

  /**
     @short to be called once all hits have been counted
  */
  inline void analyze()
  {
    nEvents_++;
    
    int maxCounts(-1);
    hotWaferKey_=UVKey_t(0,0);
    for(std::map<UVKey_t,int>::iterator it = countMap_.begin();
        it != countMap_.end();
        it++) {
      int cts(it->second);
      countH_->Fill(cts);
      if(cts<maxCounts) continue;
      hotWaferKey_=it->first;
      maxCounts=cts;
    }
    maxCountH_->Fill(maxCounts);
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
  }

  /**
     @short return true if UV is a neighboring cell
   */
  inline bool isNeighbor(UVKey_t waferKey)
  {
    int deltau(waferKey.first-myUV_.first);
    int deltav(waferKey.second-myUV_.second);
    return isNeighbor(deltau,deltav);
  }

  inline bool isNeighbor(int deltau,int deltav) 
  {
    return ( (deltau==1 && deltav==1) || (deltau==0 && deltav==1) || (deltau==-1 && deltav==0)
             || (deltau==-1 && deltav==-1) || (deltau==0 && deltav==-1) || (deltau==1 && deltav==0) );
  }  

  /**
     @short returns neighboring counts if found, otherwise -1 is returned
  */
  inline int getNeighborCounts(UVKey_t waferKey) {
    int counts(-1);
    for(std::map<UVKey_t,int>::iterator it=countMap_.begin();
        it!=countMap_.end();
        it++){
      int deltau(it->first.first-waferKey.first);
      int deltav(it->first.second-waferKey.second);
      if(!isNeighbor(deltau,deltav)) continue;
      counts=it->second;
    }
    return counts;
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
     @short returns hot wafer key data
   */
  inline const UVKey_t getHotWaferUV() { return hotWaferKey_; }
  inline const int getHotWaferCounts() { return countMap_[hotWaferKey_]; }

  /**
     @short returns cell data
   */
  inline const UVKey_t getWaferEquivalentUV() { return myUV_; }
  inline const int getWaferEquivalentCounts() { return countMap_[myUV_]; }

  inline int getCells() { return ncells_; }

  /**
     @short DTOR
   */
  ~WaferOccupancyHisto() {}

 
 private:
  int ncells_;
  TH1F *adcH_,*countH_,*maxCountH_;

  int nEvents_;
  std::map<UVKey_t,int> countMap_;
  UVKey_t myUV_,hotWaferKey_;
};

#endif
