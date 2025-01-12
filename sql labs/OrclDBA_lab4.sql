--statements from lecture 

select UPPER(last_name) from employees;

select employee_id as Even_number, last_name
from employees
where mod(employee_id, 2) = 0
order by last_name;

select last_name, to_char(hire_date, 'DD-MM-RRRR')
from employees;

select sessiontimezone, sysdate from dual;

select last_name, (sysdate-hire_date)/7 as Weeks from employees where department_id = 90;

--practice
select sysdate as "Date" 
from dual;

select employee_id, last_name, salary, salary+salary*0.155 "New salary" 
from employees;

select employee_id, last_name, salary, salary+salary*0.155 "New salary", salary*0.155 "Increase" 
from employees;

select initcap(last_name) "Name", length(last_name) "Length"
from employees 
where last_name like 'A%' or last_name like 'M%'
order by "Name";

select initcap(last_name) "Name", length(last_name) "Length"
from employees
where last_name like '&Letter%'
order by "Name";

select initcap(last_name) "Name", length(last_name) "Length"
from employees
where last_name like UPPER('&Letter%')
order by "Name";

select last_name, round(months_between(sysdate, hire_date)) as Month_worked
from employees
order by Month_worked;

select last_name, LPAD(salary, 15, '$') Salary
from employees;

select last_name,  RPAD(' ', FLOOR(salary / 1000), '*') AS salaries_in_asterisk
from employees
order by salary desc;

select last_name, trunc((sysdate- hire_date)/7,0) as TENURE
from employees
where department_id = 90
order by TENURE desc;