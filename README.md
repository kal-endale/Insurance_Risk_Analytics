# Insurance_Risk_Analytics
KAIM-week3-Insurance_Risk_Analytics

%%writefile README.md
# End-to-End Insurance Risk Analytics & Predictive Modeling

## Overview
This project focuses on diving into real insurance data to uncover low-risk segments and build smart models that optimize premiums. It covers designing A/B tests, versioning data pipelines, and engineering risk-based pricing models.

## Business Need
As a Marketing Analytics Engineer at AlphaCare Insurance Solutions (ACIS), a forward-thinking insurance company, this project aims to develop cutting-edge risk and predictive analytics for car insurance planning and marketing in South Africa.

ACIS is preparing for an aggressive growth phase and requires evidence-driven strategies to optimize marketing investments and refine pricing models, moving towards analytics-driven decisions based on historical claim data, statistical rigor, and machine learning.

## Situational Overview
The primary task is to analyze 18 months of historical insurance claim data (Feb 2014 – Aug 2015) to optimize the marketing strategy and identify "low-risk" targets. By reducing premiums for these segments, ACIS can attract new clients.

### Project Deliverables:
1.  **Understand Insurance Terminology and Risk Metrics**: Gain a deep understanding of industry-specific terms.
2.  **Statistically Validate Hypotheses**: Confirm or reject hypotheses regarding risk drivers across provinces, zip codes, and gender to inform a new segmentation strategy.
3.  **Develop Predictive Models**: Create models to estimate claim severity and probability for a dynamic, risk-based pricing system.
4.  **Communicate Findings**: Present analysis, findings, and recommendations in a clear, business-facing report for ACIS leadership.

## Data
The historical data covers car-insurance policy, client, vehicle, and claim information for ACIS from February 2014 to August 2015. Key data groups include:
-   **Policy**: UnderwrittenCoverID, PolicyID
-   **Transaction**: TransactionMonth
-   **Client**: IsVATRegistered, Citizenship, LegalType, Title, Language, Bank, AccountType, MaritalStatus, Gender
-   **Location**: Country, Province, PostalCode, MainCrestaZone, SubCrestaZone
-   **Vehicle**: ItemType, Mmcode, VehicleType, RegistrationYear, Make, Model, Cylinders, Cubiccapacity, Kilowatts, Bodytype, NumberOfDoors, VehicleIntroDate, CustomValueEstimate, AlarmImmobiliser, TrackingDevice, CapitalOutstanding, NewVehicle, WrittenOff, Rebuilt, Converted, CrossBorder, NumberOfVehiclesInFleet
-   **Plan**: SumInsured, TermFrequency, CalculatedPremiumPerTerm, ExcessSelected, CoverCategory, CoverType, CoverGroup, Section, Product, StatutoryClass, StatutoryRiskType
-   **Payment & Claim**: TotalPremium, TotalClaims

### Derived Metrics:
-   **Loss Ratio** = TotalClaims / TotalPremium
-   **Margin** = TotalPremium − TotalClaims
