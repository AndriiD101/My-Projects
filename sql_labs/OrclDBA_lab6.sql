select * from employees;

select * from employees order by last_name;

select max(salary) from employees;

select min(last_name), max(last_name) from employees;

SELECT COUNT(DISTINCT department_id) AS distinct_departments
FROM employees;

SELECT department_id, job_id, '$' || ROUND(AVG(salary), 2) as AVG_SALARY
FROM employees
GROUP BY department_id, job_id
ORDER BY department_id;

--practice
--1) TRUE
--2) FALSE
--3) FALSE 

SELECT ROUND(MAX(salary), 0) as "Maximum", ROUND(MIN(salary), 0) as "Minimum", ROUND(SUM(salary), 0) as "Sum", ROUND(AVG(salary), 0) as "Average"
FROM employees;

SELECT job_id, ROUND(MAX(salary), 0) as "Maximum", ROUND(MIN(salary), 0) as "Minimum", ROUND(SUM(salary), 0) as "Sum", ROUND(AVG(salary), 0) as "Average"
FROM employees
GROUP BY job_id;
--ORDER BY job_id;

select job_id, COUNT(*)
from employees
WHERE job_id = '&job_id'
GROUP BY job_id;

select COUNT(DISTINCT(manager_id)) as "Number of Managers"
from employees;

select max(salary) - min(salary) as Difference
from employees;

SELECT manager_id, MIN(salary) MIN_SALARY
from employees
WHERE manager_id is not NULL
GROUP BY manager_id
HAVING MIN(salary)>6000
ORDER BY MIN_SALARY DESC;

--extra challenges
SELECT 
    COUNT(*) AS TOTAL,
    SUM(CASE WHEN EXTRACT(YEAR FROM HIRE_DATE) = 2009 THEN 1 ELSE 0 END) AS "2009",
    SUM(CASE WHEN EXTRACT(YEAR FROM HIRE_DATE) = 2010 THEN 1 ELSE 0 END) AS "2010",
    SUM(CASE WHEN EXTRACT(YEAR FROM HIRE_DATE) = 2011 THEN 1 ELSE 0 END) AS "2011",
    SUM(CASE WHEN EXTRACT(YEAR FROM HIRE_DATE) = 2012 THEN 1 ELSE 0 END) AS "2012"
FROM 
    employees;

SELECT 
    JOB_ID AS Job,
    SUM(CASE WHEN DEPARTMENT_ID = 20 THEN SALARY ELSE NULL END) AS "Dept 20",
    SUM(CASE WHEN DEPARTMENT_ID = 50 THEN SALARY ELSE NULL END) AS "Dept 50",
    SUM(CASE WHEN DEPARTMENT_ID = 80 THEN SALARY ELSE NULL END) AS "Dept 80",
    SUM(CASE WHEN DEPARTMENT_ID = 90 THEN SALARY ELSE NULL END) AS "Dept 90",
    SUM(SALARY) AS Total
FROM 
    employees
GROUP BY 
    JOB_ID
ORDER BY 
    JOB_ID;
    

