select * from employees;
select * from departments;
select * from locations where city='Toronto';

select l.city, d.department_name
from locations l join departments d
using (location_id);

select e.employee_id, e.last_name, e.department_id,
        d.department_id, d.location_id
from employees e join departments d
on (e.department_id = d.department_id);

select e.last_name, e.salary, j.grade_level
from employees e join job_grades j
on e.salary between j.lowest_sal and j.highest_sal;

--practice
select l.location_id, l.street_address, l.city, l.state_province, c.country_name
from locations l
natural join countries c
order by location_id;

select e.last_name, e.department_id, d.department_name
from employees e join departments d
on (e.department_id = d.department_id);

select e.last_name, e.job_id, e.department_id, d.department_name
from employees e join departments d
on (e.department_id = d.department_id)
join locations l
using (location_id)
where l.city = 'Toronto';

select w.last_name "Employee", w.employee_id "Emp#", m.last_name "Manager", m.manager_id "Man#"
from employees w join employees m
on (w.employee_id = m.manager_id);

select w.last_name "Employee", w.employee_id "Emp#", m.last_name "Manager", m.manager_id "Man#"
from employees w right join employees m
on (w.employee_id = m.manager_id);

select e.last_name, e.salary, d.department_name, j.grade_level
from employees e 
join departments d
on (e.department_id = d.department_id)
join job_grades j
on e.salary between j.lowest_sal and j.highest_sal;