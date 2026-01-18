from typing import Dict, Any, List
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ClinicalInterpreter:
    """Generate clinical interpretation from vessel features"""

    def __init__(self, config):
        self.config = config
        self._setup_thresholds()

    def _setup_thresholds(self):
        """Setup clinical interpretation thresholds"""
        self.thresholds = {
            # Vessel Density
            "density": {
                "very_low": 0.03,
                "low": 0.05,
                "normal_low": 0.08,
                "normal_high": 0.15,
                "high": 0.20,
                "very_high": 0.25
            },
            # Vessel Width
            "width": {
                "very_thin": 1.0,
                "thin": 1.5,
                "normal_low": 2.0,
                "normal_high": 3.5,
                "wide": 4.0,
                "very_wide": 5.0
            },
            # Tortuosity
            "tortuosity": {
                "very_straight": 1.0,
                "straight": 1.2,
                "normal": 1.5,
                "mild": 1.8,
                "moderate": 2.0,
                "severe": 2.5
            },
            # Branching
            "branching": {
                "very_low": 50,
                "low": 100,
                "normal_low": 150,
                "normal_high": 350,
                "high": 400,
                "very_high": 500
            }
        }

    def interpret(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate clinical interpretation from features

        Returns:
            Dictionary with clinical interpretation
        """
        try:
            findings = []
            severity_scores = []

            # 1. Vessel Density Analysis
            density_score, density_findings = self._analyze_density(features)
            findings.extend(density_findings)
            severity_scores.append(density_score * 0.3)  # 30% weight

            # 2. Vessel Width Analysis
            width_score, width_findings = self._analyze_width(features)
            findings.extend(width_findings)
            severity_scores.append(width_score * 0.2)  # 20% weight

            # 3. Tortuosity Analysis
            tortuosity_score, tortuosity_findings = self._analyze_tortuosity(features)
            findings.extend(tortuosity_findings)
            severity_scores.append(tortuosity_score * 0.2)  # 20% weight

            # 4. Branching Analysis
            branching_score, branching_findings = self._analyze_branching(features)
            findings.extend(branching_findings)
            severity_scores.append(branching_score * 0.15)  # 15% weight

            # 5. Regional Analysis
            regional_score, regional_findings = self._analyze_regional(features)
            findings.extend(regional_findings)
            severity_scores.append(regional_score * 0.15)  # 15% weight

            # Calculate overall severity
            total_score = sum(severity_scores)
            severity = self._determine_severity(total_score)

            # Generate recommendations
            recommendations = self._generate_recommendations(severity, features)

            # Generate differentials
            differentials = self._generate_differentials(features)

            # Determine follow-up
            followup = self._determine_followup(severity)

            # Calculate confidence
            confidence = self._calculate_confidence(features, findings)

            return {
                "severity": severity,
                "findings": findings,
                "recommendations": recommendations,
                "differentials": differentials,
                "followup": followup,
                "confidence": confidence,
                "severity_score": total_score
            }

        except Exception as e:
            logger.error(f"Clinical interpretation failed: {e}")
            return self._get_empty_interpretation()

    def _analyze_density(self, features: Dict[str, Any]) -> tuple:
        """Analyze vessel density"""
        density = features.get("vessel_density", 0)
        findings = []
        score = 0

        if density < self.thresholds["density"]["very_low"]:
            findings.append("Severely reduced overall vessel density")
            score = 4
        elif density < self.thresholds["density"]["low"]:
            findings.append("Markedly reduced vessel density")
            score = 3
        elif density < self.thresholds["density"]["normal_low"]:
            findings.append("Moderately reduced vessel density")
            score = 2
        elif density > self.thresholds["density"]["very_high"]:
            findings.append("Extremely increased vessel density (proliferative changes likely)")
            score = 4
        elif density > self.thresholds["density"]["high"]:
            findings.append("Markedly increased vessel density (possible proliferative changes)")
            score = 3
        elif density > self.thresholds["density"]["normal_high"]:
            findings.append("Mildly increased vessel density")
            score = 1
        else:
            findings.append("Normal vessel density")
            score = 0

        return score, findings

    def _analyze_width(self, features: Dict[str, Any]) -> tuple:
        """Analyze vessel width"""
        mean_width = features.get("mean_width", 0)
        thin_ratio = features.get("thin_ratio", 0)
        findings = []
        score = 0

        if mean_width < self.thresholds["width"]["very_thin"]:
            findings.append("Extremely thin vessels (severe microvascular attenuation)")
            score = 3
        elif mean_width < self.thresholds["width"]["thin"]:
            findings.append("Predominantly thin vessels (microvascular pattern)")
            score = 2
        elif mean_width > self.thresholds["width"]["very_wide"]:
            findings.append("Extremely widened vessels (severe hypertensive changes)")
            score = 4
        elif mean_width > self.thresholds["width"]["wide"]:
            findings.append("Widened vessels (possible hypertensive changes)")
            score = 3
        elif thin_ratio > 0.7:
            findings.append("High proportion of thin vessels")
            score = 1
        else:
            findings.append("Normal vessel width distribution")
            score = 0

        return score, findings

    def _analyze_tortuosity(self, features: Dict[str, Any]) -> tuple:
        """Analyze vessel tortuosity"""
        tortuosity = features.get("tortuosity", 1)
        findings = []
        score = 0

        if tortuosity > self.thresholds["tortuosity"]["severe"]:
            findings.append("Severely increased vessel tortuosity")
            score = 4
        elif tortuosity > self.thresholds["tortuosity"]["moderate"]:
            findings.append("Moderately increased vessel tortuosity")
            score = 3
        elif tortuosity > self.thresholds["tortuosity"]["mild"]:
            findings.append("Mildly increased vessel tortuosity")
            score = 2
        elif tortuosity < self.thresholds["tortuosity"]["very_straight"]:
            findings.append("Unusually straight vessels")
            score = 1
        else:
            findings.append("Normal vessel tortuosity")
            score = 0

        return score, findings

    def _analyze_branching(self, features: Dict[str, Any]) -> tuple:
        """Analyze branching patterns"""
        branching = features.get("branching_points", 0)
        findings = []
        score = 0

        if branching < self.thresholds["branching"]["very_low"]:
            findings.append("Severely reduced branching (marked ischemic changes)")
            score = 4
        elif branching < self.thresholds["branching"]["low"]:
            findings.append("Reduced branching pattern (possible ischemic changes)")
            score = 3
        elif branching > self.thresholds["branching"]["very_high"]:
            findings.append("Extremely complex branching (severe proliferative changes)")
            score = 4
        elif branching > self.thresholds["branching"]["high"]:
            findings.append("Increased branching complexity (proliferative pattern)")
            score = 3
        else:
            findings.append("Normal branching pattern")
            score = 0

        return score, findings

    def _analyze_regional(self, features: Dict[str, Any]) -> tuple:
        """Analyze regional distribution"""
        central = features.get("central_density", 0)
        peripheral = features.get("peripheral_density", 0)
        findings = []
        score = 0

        if central > 0 and peripheral > 0:
            ratio = central / peripheral

            if ratio < 0.4:
                findings.append("Severe central rarefaction with peripheral dominance")
                score = 4
            elif ratio < 0.6:
                findings.append("Marked peripheral dominance with central rarefaction")
                score = 3
            elif ratio < 0.8:
                findings.append("Peripheral dominance")
                score = 1
            elif ratio > 1.6:
                findings.append("Severe central predominance")
                score = 3
            elif ratio > 1.4:
                findings.append("Marked central predominance")
                score = 2
            elif ratio > 1.2:
                findings.append("Central predominance")
                score = 1
            else:
                findings.append("Balanced regional distribution")
                score = 0
        else:
            findings.append("Insufficient data for regional analysis")
            score = 0

        return score, findings

    def _determine_severity(self, score: float) -> str:
        """Determine overall severity"""
        if score >= 3.5:
            return "Severe"
        elif score >= 2.5:
            return "Moderate"
        elif score >= 1.5:
            return "Mild"
        elif score >= 0.5:
            return "Borderline"
        else:
            return "Normal"

    def _generate_recommendations(self, severity: str, features: Dict[str, Any]) -> List[str]:
        """Generate clinical recommendations"""
        recommendations = []

        # Base recommendations
        if severity == "Normal":
            recommendations.append("Routine follow-up as per standard screening guidelines (12-24 months)")

        elif severity == "Borderline":
            recommendations.append("Repeat imaging in 6-12 months")
            recommendations.append("Consider basic vascular risk factor assessment (blood pressure, glucose)")
            recommendations.append("Lifestyle modification counseling")

        elif severity == "Mild":
            recommendations.append("Comprehensive ophthalmological examination recommended")
            recommendations.append("Systemic blood pressure and glucose screening")
            recommendations.append("Repeat imaging in 3-6 months")
            recommendations.append("Cardiovascular risk assessment")

        elif severity == "Moderate":
            recommendations.append("Urgent ophthalmology referral recommended")
            recommendations.append("Complete systemic evaluation (HTN, DM, cardiovascular)")
            recommendations.append("Consider OCT angiography for detailed assessment")
            recommendations.append("Repeat imaging in 1-3 months")
            recommendations.append("Consider lipid profile and renal function tests")

        elif severity == "Severe":
            recommendations.append("Immediate ophthalmology consultation required")
            recommendations.append("Emergency systemic medical workup")
            recommendations.append("Advanced retinal imaging (OCT-A, fluorescein angiography)")
            recommendations.append("Close monitoring with weekly follow-up")
            recommendations.append("Consider hospitalization for severe cases")

        # Feature-specific recommendations
        if features.get("tortuosity", 1) > 2.0:
            recommendations.append("Evaluate for systemic hypertension and cardiovascular disease")

        if features.get("vessel_density", 0) < 0.05:
            recommendations.append("Consider evaluation for vascular insufficiency and ischemic disease")

        if features.get("branching_points", 0) > 400:
            recommendations.append("Screen for diabetic retinopathy with dilated fundus exam")

        if features.get("mean_width", 0) > 4.0:
            recommendations.append("Evaluate for hypertensive retinopathy and cardiovascular risk")

        return list(dict.fromkeys(recommendations))  # Remove duplicates

    def _generate_differentials(self, features: Dict[str, Any]) -> List[str]:
        """Generate differential diagnosis"""
        differentials = []

        density = features.get("vessel_density", 0)
        branching = features.get("branching_points", 0)
        tortuosity = features.get("tortuosity", 1)
        mean_width = features.get("mean_width", 0)
        thin_ratio = features.get("thin_ratio", 0)
        thick_ratio = features.get("thick_ratio", 0)

        # Pattern-based differentials
        if density < 0.08 and branching < 150:
            differentials.append("Vascular insufficiency / Ischemic retinopathy")
            differentials.append("Ocular ischemic syndrome")

        if tortuosity > 1.8 and mean_width > 3.5:
            differentials.append("Hypertensive retinopathy")
            differentials.append("Cardiovascular disease")

        if density > 0.15 and branching > 350:
            differentials.append("Proliferative diabetic retinopathy")
            differentials.append("Retinal vein occlusion with neovascularization")

        if thin_ratio > 0.7:
            differentials.append("Radiation retinopathy")
            differentials.append("Retinopathy of prematurity")

        if thick_ratio > 0.3:
            differentials.append("Venous stasis retinopathy")
            differentials.append("Hyperviscosity syndromes")

        # Add general differentials if less than 3
        if len(differentials) < 3:
            general = [
                "Diabetic retinopathy",
                "Hypertensive retinopathy",
                "Retinal vein occlusion",
                "Retinal artery occlusion"
            ]
            differentials.extend(general[:3 - len(differentials)])

        return differentials[:5]  # Return top 5

    def _determine_followup(self, severity: str) -> str:
        """Determine follow-up timeframe"""
        followup_map = {
            "Normal": "12-24 months",
            "Borderline": "6-12 months",
            "Mild": "3-6 months",
            "Moderate": "1-3 months",
            "Severe": "Immediate (within 1 week)"
        }
        return followup_map.get(severity, "Consult specialist")

    def _calculate_confidence(self, features: Dict[str, Any], findings: List[str]) -> float:
        """Calculate confidence score"""
        confidence = 0.7  # Base confidence

        # Increase based on feature quality
        if features.get("vessel_density", 0) > 0.02:
            confidence += 0.1

        if features.get("branching_points", 0) > 10:
            confidence += 0.1

        if len(findings) > 0 and len(findings) <= 5:
            confidence += 0.1

        # Decrease for extreme values
        if features.get("vessel_density", 0) > 0.3:
            confidence -= 0.1

        if features.get("tortuosity", 1) > 3.0:
            confidence -= 0.1

        return min(max(confidence, 0.0), 1.0)

    def _get_empty_interpretation(self) -> Dict[str, Any]:
        """Return empty interpretation"""
        return {
            "severity": "Unknown",
            "findings": ["Unable to generate clinical interpretation"],
            "recommendations": ["Consult with retinal specialist"],
            "differentials": ["Consider comprehensive ophthalmic evaluation"],
            "followup": "Consult specialist",
            "confidence": 0.0,
            "severity_score": 0.0
        }