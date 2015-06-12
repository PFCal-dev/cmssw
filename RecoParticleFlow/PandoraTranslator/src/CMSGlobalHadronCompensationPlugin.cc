/**
 *  @file   RecoParticleFlow/PandoraTranslator/src/CMSGlobalHadronCompensationPlugin.cc
 * 
 *  @brief  Implementation of the Global Hadronic Energy Compensation strategy
 * 
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "LCHelpers/ReclusterHelper.h"

#include "RecoParticleFlow/PandoraTranslator/interface/CMSGlobalHadronCompensationPlugin.h"

#include <DataFormats/ParticleFlowReco/interface/PFRecHit.h>

#include <algorithm>

using namespace pandora;
using namespace cms_content;

//------------------------------------------------------------------------------------------------------------------------------------------

GlobalHadronCompensation::GlobalHadronCompensation() :
  m_CorrectionLevel(TRIVIAL), m_nMIPsCut(1.0), m_e_em_EE(1.0), m_e_em_FH(1.0), m_e_em_BH(1.0)
{
  
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode GlobalHadronCompensation::MakeEnergyCorrections(const Cluster *const pCluster, float &correctedHadronicEnergy) const
{
  //
  //TRIVIAL corrections are my favourite
  //
  if(m_CorrectionLevel==TRIVIAL) 
    {
      correctedHadronicEnergy *= 1.f;
      return STATUS_CODE_SUCCESS;
    }

  //
  // e.m. scale determination
  //
  float emen_EE(0.f), emen_FH(0.f), emen_BH(0.f);
  float avgen_mip_FH(0.f), nHits_FH(0.f);
  const OrderedCaloHitList& orderedCaloHitList = pCluster->GetOrderedCaloHitList();
  OrderedCaloHitList::const_iterator layer = orderedCaloHitList.begin();
  OrderedCaloHitList::const_iterator hits_end = orderedCaloHitList.end();
  for( ; layer != hits_end; ++ layer ) 
    {
      const CaloHitList& hits_in_layer = *(layer->second);
      for( const auto& hit : hits_in_layer ) 
	{
	  //get the energy in MIPs
	  float nMIPs(hit->GetMipEquivalentEnergy());

	  //check sub-detector id
	  const void* void_hit_ptr = hit->GetParentCaloHitAddress();
	  const reco::PFRecHit* original_hit_ptr = static_cast<const reco::PFRecHit*>(void_hit_ptr);
	  const uint32_t rawid = original_hit_ptr->detId();
	  const int subDetId   = (rawid>>25)&0x7;
	  
	  if(subDetId == ForwardSubdetector::HGCEE)    
	    {
	      const int layerId = (rawid>>19)&0x1f;
	      float weight( m_EEWeights.size() ? 
			    m_EEWeights[layerId-1] : 
			    1.0);
	      emen_EE += weight*nMIPs/m_e_em_EE;
	    }
	  else if(subDetId == ForwardSubdetector::HGCHEB)
	    {
	      float weight( m_BHWeights.size() ? 
			    m_BHWeights[0] : 
			    1.0);
	      emen_BH += weight*nMIPs/m_e_em_BH;
	    }
	  else if( subDetId == ForwardSubdetector::HGCHEF ) 
	    {
	      const int layerId = (rawid>>19)&0x1f;
	      float weight( m_FHWeights.size() ? 
			    (layerId>1 ? m_FHWeights[1] : m_FHWeights[0]) : 
			    1.0);
	      emen_FH     += weight*nMIPs/m_e_em_FH;
	      avgen_mip_FH += nMIPs;
	      nHits_FH     += 1.0;
	    }
	}
    }
  
  //the e.m. scale is determined
  float emen(emen_EE+emen_FH+emen_BH);

  //check if it makes sense...
  if(emen<0.01f)
    {
      correctedHadronicEnergy *= 1.f;
      return STATUS_CODE_SUCCESS;
    }

  //stop here if no further corrections are requested
  if(m_CorrectionLevel==EMWEIGHTED)
    {
      correctedHadronicEnergy=emen;
      return STATUS_CODE_SUCCESS;
    }

  //
  //pi/e correction
  //
  float pioeen_EE(getByPiOverECorrectedEn(emen_EE,emen,int(ForwardSubdetector::HGCEE))); 
  float pioeen_FH(getByPiOverECorrectedEn(emen_FH,emen,int(ForwardSubdetector::HGCHEF))); 
  float pioeen_BH(getByPiOverECorrectedEn(emen_BH,emen,int(ForwardSubdetector::HGCHEB))); 
  float pioeen_HCAL(pioeen_FH+pioeen_BH);
  float pioeen(pioeen_EE+pioeen_HCAL);

  //check if it makes sense...
  if(pioeen<0.01f)
    {
      correctedHadronicEnergy *= 1.f;
      return STATUS_CODE_SUCCESS;
    }

  //stop here if no further corrections are requested
  if(m_CorrectionLevel==PIOECORRECTED)
    {
      correctedHadronicEnergy=pioeen;
      return STATUS_CODE_SUCCESS;
    }

  
  //
  //residual corrections, depending on energy sharing
  //

  //energy sharing (ECAL MIP subtracted)
  float pioeen_m_EEmip(std::max(pioeen_EE-m_IntegMIP_emEn,0.f)+pioeen_HCAL);
  float enFracInHCAL_m_mip(-1);
  if(pioeen_HCAL==0)   enFracInHCAL_m_mip=0;
  if(pioeen_m_EEmip>0) enFracInHCAL_m_mip=pioeen_HCAL/pioeen_m_EEmip;
  
  //apply residual correction according to energy sharing
  float residualScale=getResidualScale(pioeen,enFracInHCAL_m_mip);
  float rsen=pioeen*residualScale;

  //check if it makes sense...
  if(pioeen<0.01f)
    {
      correctedHadronicEnergy *= 1.f;
      return STATUS_CODE_SUCCESS;
    }

  //stop here if no further corrections are requested
  if(m_CorrectionLevel==PIRESIDUALS)
    {
      correctedHadronicEnergy=rsen;
      return STATUS_CODE_SUCCESS;
    }


  //
  // global compensation if energy in FH is significant
  //
  float c_FH(1.0);
  if(nHits_FH>0)
    {
      avgen_mip_FH/=nHits_FH;
      layer = orderedCaloHitList.begin();
      float nHits_avg_FH(0.0), nHits_elim_FH(0.0);
      for( ; layer != hits_end; ++ layer ) 
	{
	  const CaloHitList& hits_in_layer = *(layer->second);
	  for( const auto& hit : hits_in_layer ) 
	    {
	      // hack so that we can know if we are in the HEF or not (go back to cmssw det id)                             
	      const void* void_hit_ptr = hit->GetParentCaloHitAddress();
	      const reco::PFRecHit* original_hit_ptr = static_cast<const reco::PFRecHit*>(void_hit_ptr);
	      const uint32_t rawid = original_hit_ptr->detId();
	      const int subDetId = (rawid>>25)&0x7;
	      if( subDetId != ForwardSubdetector::HGCHEF ) continue;
	      
	      const float nMIPs = hit->GetMipEquivalentEnergy();
	      nHits_avg_FH  += ( nMIPs > avgen_mip_FH);
	      nHits_elim_FH += ( nMIPs > m_nMIPsCut );
	    }
	}
      
      c_FH=(nHits_FH-nHits_elim_FH)/(nHits_FH-nHits_avg_FH);
    }

  //apply at pi/e level and rescale the residual scaled energy
  float gcen = rsen*((pioeen_EE + c_FH*pioeen_FH + pioeen_BH)/pioeen);
 
  if(gcen<0.01f)
    {
      correctedHadronicEnergy *= 1.f;
      return STATUS_CODE_SUCCESS;
    }

  //all done here
  correctedHadronicEnergy=gcen;
  return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

float GlobalHadronCompensation::GetHadronicEnergyInLayer(const OrderedCaloHitList &orderedCaloHitList, const unsigned int pseudoLayer) const
{
  OrderedCaloHitList::const_iterator iter = orderedCaloHitList.find(pseudoLayer);

  float hadronicEnergy(0.f);

  if (iter != orderedCaloHitList.end())
    {
      for (CaloHitList::const_iterator hitIter = iter->second->begin(), hitIterEnd = iter->second->end(); hitIter != hitIterEnd; ++hitIter)
        {
	  hadronicEnergy += (*hitIter)->GetHadronicEnergy();
        }
    }

  return hadronicEnergy;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode GlobalHadronCompensation::ReadSettings(const TiXmlHandle xmlHandle)
{

  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "MipEnergyThreshold", m_nMIPsCut));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "IntegMIP_emEn",      m_IntegMIP_emEn));

  float correctionLevel(0.0);
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "CorrectionLevel", correctionLevel));
  m_CorrectionLevel=(int)correctionLevel;

  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "EEWeights", m_EEWeights));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "FHWeights", m_FHWeights));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "BHWeights", m_BHWeights));

  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "emEE", m_e_em_EE));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "emFH", m_e_em_FH));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "emBH", m_e_em_BH));

  std::cout << "[CMSGlobalHadronCompensationPlugin::ReadSettings]" << std::endl
	    << "MIP threshold for GC: " << m_nMIPsCut << std::endl
	    << "MIP value in EE (after pi/e/):" << m_IntegMIP_emEn << std::endl
	    << "e.m. scales " << 1./m_e_em_EE << " " << 1./m_e_em_FH << " " << 1./m_e_em_BH << std::endl
	    << "# layer weights " << m_EEWeights.size() << " " << m_FHWeights.size() << " " << m_BHWeights.size() << std::endl;

  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "pioeEE", m_pioe_EE));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "pioeFH", m_pioe_FH));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "pioeBH", m_pioe_BH));

  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "pioebananaParam0", m_pioe_bananaParam_0 ));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "pioebananaParam1", m_pioe_bananaParam_1 ));
  PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "pioebananaParam2", m_pioe_bananaParam_2 ));

  std::cout << "# pi/e parameters " << m_pioe_EE.size() << " " << m_pioe_FH.size() << " " << m_pioe_BH.size() <<" " << std::endl
	    << "# res. corr parameters " << m_pioe_bananaParam_0.size() << " " << m_pioe_bananaParam_1.size()<< " " << m_pioe_bananaParam_2.size() << std::endl;
  
  return STATUS_CODE_SUCCESS;
}
