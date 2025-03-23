select * from employees;

select last_name, salary 
from employees 
where salary > 12000;

select last_name, department_id 
from employees 
where employee_id = 176;

select last_name, salary 
from employees 
where salary < 5000 or salary > 12000;

--select last_name, salary 
--from employees 
--where salary not between 5000 and 12000;

select last_name, job_id, hire_date 
from employees 
where last_name = 'Taylor' or last_name = 'Matos'
order by hire_date;

select last_name, department_id 
from employees 
where department_id in (20, 50) order by last_name;

select last_name "Employee", salary "Month salary" 
from employees 
where salary between 5000 and 12000 and department_id in (20, 50) 
order by "Month salary" desc;

select last_name, hire_date 
from employees 
where extract(year from hire_date) = 2010;

select last_name, job_id 
from employees 
where manager_id is NULL;

select last_name, salary, commission_pct 
from employees 
where commission_pct is not null 
order by 2 desc, 3 desc;

select last_name, salary 
from employees 
where salary > &salary;

select employee_id, last_name, salary, department_id 
from employees 
where manager_id=&manager_id 
order by &order_comlumn;

--bonus tasks

select last_name 
from employees 
where last_name like '__a%';

select last_name 
from employees 
where last_name like '%a%' and last_name like '%e%';

--extra challenges

select last_name, job_id, salary 
from employees 
where job_id in ('SA_REP' , 'ST_CLERK') and salary not in (2500, 3500,  7000) 
order by salary desc;

select last_name "Employee", salary "Month salary", commission_pct 
from employees 
where commission_pct = 0.2;















drop table employees;

CREATE TABLE employees (
    employee_id NUMBER PRIMARY KEY,
    first_name VARCHAR2(50),
    last_name VARCHAR2(50),
    email VARCHAR2(100),
    phone_number VARCHAR2(50),
    hire_date DATE,
    job_id VARCHAR2(10),
    salary NUMBER,
    commission_pct NUMBER(3, 2),
    manager_id NUMBER,
    department_id NUMBER
);

INSERT INTO employees VALUES (100, 'Steven', 'King', 'SKING', '515.123.4567', TO_DATE('17-JUN-11', 'DD-MON-YY'), 'AD_PRES', 24000, NULL, NULL, 90);
INSERT INTO employees VALUES (101, 'Neena', 'Kochar', 'NKOCHAR', '515.123.4568', TO_DATE('21-SEP-09', 'DD-MON-YY'), 'AD_VP', 17000, NULL, 100, 90);
INSERT INTO employees VALUES (102, 'Lex', 'De Haan', 'LDEHAAN', '515.123.4569', TO_DATE('13-JAN-09', 'DD-MON-YY'), 'AD_VP', 17000, NULL, 100, 90);
INSERT INTO employees VALUES (103, 'Alexander', 'Hunold', 'AHUNOLD', '590.423.4567', TO_DATE('03-JAN-14', 'DD-MON-YY'), 'IT_PROG', 9000, NULL, 102, 60);
INSERT INTO employees VALUES (104, 'Bruce', 'Ernst', 'BERNST', '590.423.4568', TO_DATE('07-MAY-15', 'DD-MON-YY'), 'IT_PROG', 6000, NULL, 103, 60);
INSERT INTO employees VALUES (107, 'Diana', 'Lorentz', 'DLORENTZ', '590.423.5567', TO_DATE('07-FEB-15', 'DD-MON-YY'), 'IT_PROG', 4200, NULL, 103, 60);
INSERT INTO employees VALUES (124, 'Kevin', 'Mourgos', 'KMOURGOS', '650.123.5234', TO_DATE('16-NOV-15', 'DD-MON-YY'), 'ST_MAN', 5800, NULL, 100, 50);
INSERT INTO employees VALUES (141, 'Trenna', 'Rajs', 'TRAJS', '650.121.8669', TO_DATE('17-OCT-11', 'DD-MON-YY'), 'ST_CLERK', 3500, NULL, 124, 50);
INSERT INTO employees VALUES (142, 'Curtis', 'Davies', 'CDAVIES', '650.121.2994', TO_DATE('29-JAN-13', 'DD-MON-YY'), 'ST_CLERK', 3100, NULL, 124, 50);
INSERT INTO employees VALUES (143, 'Randall', 'Matos', 'RMATOS', '650.121.2874', TO_DATE('15-MAR-14', 'DD-MON-YY'), 'ST_CLERK', 2600, NULL, 124, 50);
INSERT INTO employees VALUES (144, 'Peter', 'Vargas', 'PVARGAS', '650.121.2004', TO_DATE('09-JUL-14', 'DD-MON-YY'), 'ST_CLERK', 2500, NULL, 124, 50);
INSERT INTO employees VALUES (149, 'Eleni', 'Zlotkey', 'EZLOTKEY', '011.44.1344.429018', TO_DATE('29-JAN-16', 'DD-MON-YY'), 'SA_MAN', 10500, 0.2, 100, 80);
INSERT INTO employees VALUES (174, 'Ellen', 'Abel', 'EABEL', '011.44.1344.429267', TO_DATE('11-MAY-12', 'DD-MON-YY'), 'SA_REP', 11000, 0.3, 149, 80);
INSERT INTO employees Values (176, 'Jonathan', 'Taylor', 'JTAYLOR', '011.44.164.429265', TO_DATE('24-MAR-14' ,'DD-MON-YY'), 'SA_REP', 8600, 0.2, 149, 80);
INSERT INTO employees VALUES (178, 'Kimberly', 'Grant', 'KGRANT', '011.44.1344.429263', TO_DATE('24-MAY-15', 'DD-MON-YY'), 'SA_REP', 7000, 0.15, 149, 80);
INSERT INTO employees VALUES (200, 'Jennifer', 'Whalen', 'JWHALEN', '515.123.4444', TO_DATE('17-SEP-11', 'DD-MON-YY'), 'AD_ASST', 6000, NULL, 101, 10);
INSERT INTO employees VALUES (201, 'Michael', 'Hartstein', 'MHARTSTE', '515.123.5555', TO_DATE('17-FEB-12', 'DD-MON-YY'), 'MK_MAN', 13000, NULL, 100, 20);
INSERT INTO employees VALUES (202, 'Pat', 'Fay', 'PFAY', '603.123.6666', TO_DATE('17-AUG-13', 'DD-MON-YY'), 'MK_REP', 6000, NULL, 201, 20);
INSERT INTO employees VALUES (205, 'Shelley', 'Higgins', 'SHIGGINS', '515.123.8888', TO_DATE('07-JUN-10', 'DD-MON-YY'), 'AC_MGR', 12000, NULL, 101, 110);
INSERT INTO employees VALUES (206, 'William', 'Gietz', 'WGIETZ', '515.123.8181', TO_DATE('07-JUN-10', 'DD-MON-YY'), 'AC_ACCOUNT', 8300, NULL, 205, 110);
