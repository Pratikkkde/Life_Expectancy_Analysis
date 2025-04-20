create database life_expectancy_analysis;
use life_expectancy_analysis;
Create table life_expectancy (
country VARCHAR(100),
    year int,
    status varchar(20),
    life_expectancy float,
    adult_mortality float,
    infant_deaths float,
    alcohol float,
    percentage_expenditure float,
    hepatitis_b float,
    measles float,
    bmi float,
    under_five_deaths float,
    polio float,
    total_expenditure float,
    diphtheria float,
    hiv_aids float,
    gdp float,
    population float,
    thinness_10_19_years float,
    thinness_5_9_years float,
    income_composition_of_resources float,
    schooling float,
    predicted_life_expectancy float,
    error float
);

-- Avg. life expectancy per country
SELECT country, ROUND(AVG(life_expectancy), 2) AS avg_life_expectancy
FROM life_expectancy
GROUP BY country
ORDER BY avg_life_expectancy DESC;

-- Comparision of countries based on status
SELECT status, 
ROUND(AVG(life_expectancy), 2) AS avg_life,
ROUND(AVG(income_composition_of_resources), 2) AS avg_income_index,
ROUND(AVG(gdp), 2) AS avg_gdp
FROM life_expectancy
GROUP BY status;

-- Variation in life expectancy based on alcohol consumption statistics
SELECT ROUND(alcohol, 1) AS alcohol_level,
       ROUND(AVG(life_expectancy), 2) AS avg_life_expectancy
FROM life_expectancy
GROUP BY alcohol_level
ORDER BY alcohol_level;

-- Effects of varied gdp on deaths of kids under the age of 5
SELECT ROUND(gdp, -3) AS gdp_range,
       ROUND(AVG(under_five_deaths), 2) AS avg_under5_deaths
FROM life_expectancy
WHERE gdp IS NOT NULL
GROUP BY gdp_range
ORDER BY gdp_range;

-- Avg life expectancy based on country and year
SELECT country, year, ROUND(AVG(life_expectancy), 2) AS avg_life
FROM life_expectancy
GROUP BY country, year
ORDER BY country, year;

-- Countries with declining life expectancy
SELECT a.country, a.year, a.life_expectancy AS current, b.life_expectancy AS previous
FROM life_expectancy a
JOIN life_expectancy b ON a.country = b.country AND a.year = b.year + 1
WHERE a.life_expectancy < b.life_expectancy;

-- Impact of healthcare factors on life expectancy
SELECT 
ROUND(AVG(life_expectancy), 2) AS avg_life,
ROUND(AVG(percentage_expenditure), 2) AS health_spending,
ROUND(AVG(diphtheria), 2) AS diphtheria_rate,
ROUND(AVG(hiv_aids), 2) AS hiv_deaths,
ROUND(AVG(schooling), 2) AS avg_schooling
FROM life_expectancy;

-- Identification of high risk countries based on life expectancy mortality and GDP
SELECT country, year, life_expectancy, adult_mortality, gdp
FROM life_expectancy
WHERE life_expectancy < 60 AND adult_mortality > 250 AND gdp < 5000
ORDER BY life_expectancy;

-- Countires that spend alot on healthcare yet have lesser life expectancy
SELECT country, 
ROUND(AVG(total_expenditure), 2) AS avg_spending,
ROUND(AVG(life_expectancy), 2) AS avg_life,
ROUND(AVG(life_expectancy) / AVG(total_expenditure), 2) AS efficiency_score
FROM life_expectancy
GROUP BY country
ORDER BY efficiency_score ASC
LIMIT 10;

-- Countries that spend the least on healthcare and has the highest life expectancy
SELECT country,
ROUND(AVG(total_expenditure), 2) AS avg_spending,
ROUND(AVG(life_expectancy), 2) AS avg_life,
ROUND(AVG(life_expectancy) / AVG(total_expenditure), 2) AS efficiency_score
FROM life_expectancy
WHERE total_expenditure IS NOT NULL AND life_expectancy IS NOT NULL
GROUP BY country
ORDER BY efficiency_score DESC
LIMIT 10;

-- Life Expectancy based on the gdp group and average income index of the country
SELECT 
    ROUND(gdp, -3) AS gdp_group,
    ROUND(AVG(income_composition_of_resources), 2) AS avg_income_index,
    ROUND(AVG(life_expectancy), 2) AS avg_life
FROM life_expectancy
GROUP BY gdp_group
ORDER BY gdp_group;

-- Model accuracy summary
SELECT 
    ROUND(AVG(error), 2) AS avg_error,
    ROUND(AVG(ABS(error)), 2) AS avg_absolute_error,
    ROUND(MAX(ABS(error)), 2) AS max_absolute_error
FROM life_expectancy;

-- country wise performance of the prediction model
SELECT 
    country,
    ROUND(AVG(ABS(error)), 2) AS avg_absolute_error,
    ROUND(AVG(error), 2) AS bias -- positive means underpredicting, negative means overpredicting
FROM life_expectancy
GROUP BY country
ORDER BY avg_absolute_error ASC;

-- Consistent predictive performance across many countries, especially where data availability is strong
SELECT country,
ROUND(AVG(predicted_life_expectancy), 2) AS avg_predicted_life
FROM life_expectancy
GROUP BY country
ORDER BY avg_predicted_life DESC;

-- The model effectively distinguishes between developed and developing countries, reflecting real-world patterns in health and income distribution
SELECT status,
ROUND(AVG(life_expectancy), 2) AS avg_actual_life_expectancy,
ROUND(AVG(predicted_life_expectancy), 2) AS avg_predicted_life_expectancy
FROM life_expectancy
GROUP BY status;

-- The dip in predicted life expectancy between 2013 and 2015 indicates that consistent efforts are still needed to improve global health and living conditions
SELECT year,
ROUND(AVG(predicted_life_expectancy), 2) AS predicted_life_trend
FROM life_expectancy
GROUP BY year
ORDER BY year;

-- The difference between actual and predicted life expectancy may suggest that health interventions may have either outperformed expectations or highlight areas needing further improvement
SELECT country, year, life_expectancy, predicted_life_expectancy,
ROUND(predicted_life_expectancy - life_expectancy, 2) AS model_signal
FROM life_expectancy
ORDER BY ABS(model_signal) desc
LIMIT 10;