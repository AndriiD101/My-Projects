select * from employees;
select * from departments;
undefine last_name;

select last_name, hire_date, department_id
from employees
where department_id = (select department_id
                      from employees
                      where last_name = '&last_name')
and last_name <> '&last_name'
order by last_name;

select AVG(salary) from employees;

select employee_id, last_name, salary
from employees
where salary>(select AVG(salary)
              from employees)
order by salary;

select employee_id, last_name
from employees
where department_id in (select department_id
                          from employees
                          where last_name like '%u%');

select last_name, salary
from employees
where manager_id in (select employee_id
                    from employees
                    where last_name = 'King');
                    
select department_id, last_name, job_id
from employees
where department_id in (select department_id
                       from departments
                       where department_name = 'Executive');
                       
select last_name
from employees
where salary > any(select salary
                   from employees
                   where department_id = 60);

select employee_id, last_name, salary
from employees
where salary > (select AVG(salary)
              from employees)
and department_id in (select department_id
                          from employees
                          where last_name like '%u%');
                    