select * from my_employee;

drop table my_employee;

create table my_employee(
id number not null,
last_name varchar(25),
first_name varchar(28),
userid varchar(8),
salary number(9, 2)
);

insert into my_employee(id, last_name, first_name, userid, salary) values (1, 'Patel', 'Ralph', 'rpatel', 895);
insert into my_employee values (2, 'Dancs', 'Betty', 'bdancs', 860);
insert into my_employee values (&id, '&last_name', '&first_name', '&userid', &salary);
insert into my_employee values (4, 'Newman', 'Chad', 'cnewman', 750);
insert into my_employee values (5, 'Ropeburn', 'Audrey', 'aropebur', 1550);

update my_employee set last_name = 'Drexler' where id = 3;
update my_employee set salary=1000 where salary<900;

delete my_employee where last_name='Dancs';

savepoint step_17;

delete from my_employee;

rollback to step_17;

commit;

insert into my_employee
values (&id, '&&last_name', '&&first_name', lower(substr('&first_name', 1, 1) || substr('&last_name', 1, 7)), &salary);
undefine last_name;
undefine first_name;

select * from my_employee;

commit;
