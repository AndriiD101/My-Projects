select first_name, last_name, salary
from employees
where INSTR(salary, '5') > 0;

select last_name, NVL(commission_pct, 0)
from employees;

select last_name, job_id, salary,
    DECODE(job_id, 'IT_PROG', 1.10 * salary,
                   'ST_CLERK', 1.15*salary,
                   'SA_REP', 1.20*salary,
            salary) REVISED_SALARY
from employees;

--home work
select last_name ||  ' earns' || 
TO_CHAR(salary, '$99,999.00') || 
' monthly but wants' || TO_CHAR(salary*3, '$99,999.00') ||'.' as "Dream Salaries" 
from employees;

select last_name, hire_date, 
TO_CHAR(
        NEXT_DAY(ADD_MONTHS(HIRE_DATE, 6), 'MONDAY'),
        'fmDay, "the" Ddspth "of" Month, YYYY'
    ) AS REVIEW 
from employees;

select last_name, 
case when commission_pct is null then 'no commission'
    else to_char(commission_pct) end as COMM
from employees;

select last_name, coalesce(to_char(commission_pct),'No commission') as commm from employees; 

select job_id,
    case when job_id = 'AD_PRES' then 'A'
    when job_id = 'ST_MAN' then 'B'
    when job_id = 'IT_PROG' then 'C'
    when job_id = 'SA_REP' then 'D'
    when job_id = 'ST_CLERK' then 'E'
    else '0' end as GRADE
from employees;

select * from employees;