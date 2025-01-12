select job_id from employees;
select * from countries;
select * from locations;
select * from departments;

select department_id
from departments
MINUS
select department_id
from employees
where job_id = 'ST_CLERK';

select country_id, country_name
from countries
minus
select l.country_id, c.country_name
from locations l join countries c
on (l.country_id = c.country_id)
join departments d
on d.location_id = l.location_id;

select employee_id, job_id, department_id
from employees
where department_id = 50
UNION ALL
select employee_id, job_id, department_id
from employees
where department_id = 80;

select employee_id
from employees
where job_id = 'SA_REP'
intersect
select employee_id
from employees
where department_id = 80;

select last_name, department_id, TO_CHAR(NULL) department_name
from employees
union 
select TO_CHAR(NULL) last_name, department_id, department_name
from departments;