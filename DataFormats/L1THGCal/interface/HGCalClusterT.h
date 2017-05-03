#ifndef DataFormats_L1Trigger_HGCalClusterT_h
#define DataFormats_L1Trigger_HGCalClusterT_h

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/ClusterShapes.h"
#include "Math/Vector3D.h"


namespace l1t 
{
  template <class C> class HGCalClusterT : public L1Candidate 
  {

    public:
      typedef typename edm::PtrVector<C>::const_iterator const_iterator;

    public:
      HGCalClusterT(){}
      HGCalClusterT( const LorentzVector p4,
          int pt=0,
          int eta=0,
          int phi=0
          )
        : L1Candidate(p4, pt, eta, phi),
        valid_(true),
        detId_(0),
        centre_(0, 0, 0),
        centreProj_(0., 0., 0.),
        mipPt_(0),
        seedMipPt_(0){}

      HGCalClusterT( const edm::Ptr<C>& c ):
        valid_(true)
      {
        addConstituent(c);
      }
      
      ~HGCalClusterT() {};
      
      const edm::PtrVector<C>& constituents() const {return constituents_;}        
      const_iterator constituents_begin() const {return constituents_.begin();}
      const_iterator constituents_end() const {return constituents_.end();}
      unsigned size() const { return constituents_.size(); }

      void addConstituent( const edm::Ptr<C>& c )
      {
        if( constituents_.empty() )
        { 
          detId_ = HGCalDetId(c->detId());
          seedMipPt_ = c->mipPt();
        }

        /* update cluster positions */
        Basic3DVector<float> constituentCentre( c->position() );
        Basic3DVector<float> clusterCentre( centre_ );

        clusterCentre = clusterCentre*mipPt_ + constituentCentre*c->mipPt();
        if( mipPt_ + c->mipPt()!=0 ) 
        {
          clusterCentre /= ( mipPt_ + c->mipPt() ) ;
        }
        centre_ = GlobalPoint( clusterCentre );

        if( clusterCentre.z()!=0 ) 
        {
          centreProj_= GlobalPoint( clusterCentre / clusterCentre.z() );
        }
        /* update cluster energies */
        mipPt_ += c->mipPt();

        int updatedPt = hwPt() + c->hwPt();
        setHwPt(updatedPt);

        math::PtEtaPhiMLorentzVector updatedP4 ( p4() );
        updatedP4 += c->p4(); 
        setP4( updatedP4 );

        constituents_.push_back( c );

      }
      
      bool valid() const { return valid_;}
      void setValid(bool valid) { valid_ = valid;}
      
      double mipPt() const { return mipPt_; }
      double seedMipPt() const { return seedMipPt_; }
      uint32_t detId() const { return detId_.rawId(); }

      /* distance in 'cm' */
      double distance( const l1t::HGCalTriggerCell &tc ) const 
      {
        return ( tc.position() - centre_ ).mag();
      }

      const GlobalPoint& position() const { return centre_; } 
      const GlobalPoint& centre() const { return centre_; }
      const GlobalPoint& centreProj() const { return centreProj_; }

      // FIXME: will need to fix places where the shapes are directly accessed
      // Right now keep shapes() getter as non-const 
      ClusterShapes& shapes() {return shapes_;}
      double hOverE() const
      {
        double pt_em = 0.;
        double pt_had = 0.;
        double hOe = 0.;

        for(const auto& constituent : constituents())
        {
          switch( constituent->subdetId() )
          {
            case HGCEE:
              pt_em += constituent->pt();
              break;
            case HGCHEF:
              pt_had += constituent->pt();
              break;
            case HGCHEB:
              pt_had += constituent->pt();
              break;
            default:
              break;
          }
        }
        if(pt_em>0) hOe = pt_had / pt_em ;
        else hOe = -1.;
        return hOe;
      }

      uint32_t subdetId() const {return detId_.subdetId();} 
      uint32_t layer() const {return detId_.layer();}
      int32_t zside() const {return detId_.zside();}

      int32_t ClusterET()             const {return ClusterET_;}
      int32_t ClusterCenterX()        const {return ClusterCenterX_;}
      int32_t ClusterCenterY()        const {return ClusterCenterY_;}
      int32_t ClusterSizeY()          const {return ClusterSizeY_;}
      int32_t ClusterSizeX()          const {return ClusterSizeX_;}
      int32_t ClusterNoTriggerCells() const {return ClusterNoTriggerCells_;}
      int32_t ClusterLocalMax()       const {return ClusterLocalMax_;}
      int32_t ClusterLocalMax0ET()    const {return ClusterLocalMax0ET_;}
      int32_t ClusterLocalMax0RelY()  const {return ClusterLocalMax0RelY_;}
      int32_t ClusterLocalMax0RelX()  const {return ClusterLocalMax0RelX_;}
      int32_t ClusterLocalMax1ET()    const {return ClusterLocalMax1ET_;}
      int32_t ClusterLocalMax1RelY()  const {return ClusterLocalMax1RelY_;}
      int32_t ClusterLocalMax1RelX()  const {return ClusterLocalMax1RelX_;}
      int32_t ClusterLocalMax2ET()    const {return ClusterLocalMax2ET_;}
      int32_t ClusterLocalMax2RelY()  const {return ClusterLocalMax2RelY_;}
      int32_t ClusterLocalMax2RelX()  const {return ClusterLocalMax2RelX_;}
      int32_t ClusterLocalMax3ET()    const {return ClusterLocalMax3ET_;}
      int32_t ClusterLocalMax3RelY()  const {return ClusterLocalMax3RelY_;}
      int32_t ClusterLocalMax3RelX()  const {return ClusterLocalMax3RelX_;}
      
      void SetClusterET(const int32_t CET)               { ClusterET_ = CET;}
      void SetClusterCenterX(const int32_t CCX)          { ClusterCenterX_ = CCX;}
      void SetClusterCenterY(const int32_t CCY)          { ClusterCenterY_ = CCY;}
      void SetClusterSizeY(const int32_t CSY)            { ClusterSizeY_ = CSY;}
      void SetClusterSizeX(const int32_t CSX)            { ClusterSizeX_ = CSX;}
      void SetClusterNoTriggerCells(const int32_t CNTC)  { ClusterNoTriggerCells_ = CNTC;}
      void SetClusterLocalMax(const int32_t CLM)         { ClusterLocalMax_ = CLM;}
      void SetClusterLocalMax0ET(const int32_t CLM0ET)   { ClusterLocalMax0ET_ = CLM0ET;}
      void SetClusterLocalMax0RelY(const int32_t CLM0RY) { ClusterLocalMax0RelY_ = CLM0RY;}
      void SetClusterLocalMax0RelX(const int32_t CLM0RX) { ClusterLocalMax0RelX_ = CLM0RX;}
      void SetClusterLocalMax1ET(const int32_t CLM1ET)   { ClusterLocalMax1ET_ = CLM1ET;}
      void SetClusterLocalMax1RelY(const int32_t CLM1RY) { ClusterLocalMax1RelY_ = CLM1RY;}
      void SetClusterLocalMax1RelX(const int32_t CLM1RX) { ClusterLocalMax1RelX_ = CLM1RX;}
      void SetClusterLocalMax2ET(const int32_t CLM2ET)   { ClusterLocalMax2ET_ = CLM2ET;}
      void SetClusterLocalMax2RelY(const int32_t CLM2RY) { ClusterLocalMax2RelY_ = CLM2RY;}
      void SetClusterLocalMax2RelX(const int32_t CLM2RX) { ClusterLocalMax2RelX_ = CLM2RX;}
      void SetClusterLocalMax3ET(const int32_t CLM3ET)   { ClusterLocalMax3ET_ = CLM3ET;}
      void SetClusterLocalMax3RelY(const int32_t CLM3RY) { ClusterLocalMax3RelY_ = CLM3RY;}
      void SetClusterLocalMax3RelX(const int32_t CLM3RX) { ClusterLocalMax3RelX_ = CLM3RX;}
      

      /* operators */
      bool operator<(const HGCalClusterT<C>& cl) const {return mipPt() < cl.mipPt();}
      bool operator>(const HGCalClusterT<C>& cl) const  { return  cl<*this;   }
      bool operator<=(const HGCalClusterT<C>& cl) const { return !(cl>*this); }
      bool operator>=(const HGCalClusterT<C>& cl) const { return !(cl<*this); }

    private:
        
      bool valid_;
      HGCalDetId detId_;     
      edm::PtrVector<C> constituents_;
      GlobalPoint centre_;
      GlobalPoint centreProj_; // centre projected onto the first HGCal layer


      int32_t ClusterET_            = 0;
      int32_t ClusterCenterX_       = 0;
      int32_t ClusterCenterY_       = 0;
      int32_t ClusterSizeY_         = 0;
      int32_t ClusterSizeX_         = 0;
      int32_t ClusterNoTriggerCells_= 0;
      int32_t ClusterLocalMax_      = 0;
      int32_t ClusterLocalMax0ET_   = 0;
      int32_t ClusterLocalMax0RelY_ = 0;
      int32_t ClusterLocalMax0RelX_ = 0;
      int32_t ClusterLocalMax1ET_   = 0;
      int32_t ClusterLocalMax1RelY_ = 0;
      int32_t ClusterLocalMax1RelX_ = 0;
      int32_t ClusterLocalMax2ET_   = 0;
      int32_t ClusterLocalMax2RelY_ = 0;
      int32_t ClusterLocalMax2RelX_ = 0;
      int32_t ClusterLocalMax3ET_   = 0;
      int32_t ClusterLocalMax3RelY_ = 0;
      int32_t ClusterLocalMax3RelX_ = 0;


      double mipPt_;
      double seedMipPt_;

      ClusterShapes shapes_;

  };

}

#endif
