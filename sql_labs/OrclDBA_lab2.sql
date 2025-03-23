create table Departments(
DEPARTMENT_ID number(4,0) NOT NULL,
DPARTMENT_NAME varchar(30) NOT NULL,
MANAGER_ID number(6,0),
LOCATION_ID number(4, 0)
);

insert into Departments(DEPARTMENT_ID, DPARTMENT_NAME, MANAGER_ID, LOCATION_ID) VALUES(10, 'Administaration', 200, 1700);
insert into Departments(DEPARTMENT_ID, DPARTMENT_NAME, MANAGER_ID, LOCATION_ID) VALUES(20, 'Marketing', 201, 1800);
insert into Departments(DEPARTMENT_ID, DPARTMENT_NAME, MANAGER_ID, LOCATION_ID) VALUES(50, 'Shipping', 124, 1500);
insert into Departments(DEPARTMENT_ID, DPARTMENT_NAME, MANAGER_ID, LOCATION_ID) VALUES(60, 'IT', 103, 1400);
insert into Departments(DEPARTMENT_ID, DPARTMENT_NAME, MANAGER_ID, LOCATION_ID) VALUES(80, 'Sales', 149, 2500);
insert into Departments(DEPARTMENT_ID, DPARTMENT_NAME, MANAGER_ID, LOCATION_ID) VALUES(90, 'Executive', 100, 1700);
insert into Departments(DEPARTMENT_ID, DPARTMENT_NAME, MANAGER_ID, LOCATION_ID) VALUES(110, 'Accounting', 205, 1700);
insert into Departments(DEPARTMENT_ID, DPARTMENT_NAME, MANAGER_ID, LOCATION_ID) VALUES(190, 'Contracting', NULL, 1700);

select * from Departments;
select * from Employees;
drop table Departments;

describe departments;