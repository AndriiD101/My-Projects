
create or replace view players_view as
select initcap(player_name) as "Player Name",  player_nickname, player_role, length(player_nickname) as nickname_length
from players
where player_role in ('AWPer', 'In-Game Leader');

select * from players_view;

create or replace view players_age_view as
select player_nickname, age,
    case 
        when age between 18 and 20 then 'Young'
        when age between 21 and 25 then 'Adult'
        else 'Other'
    end as age_category
from players
where age between 18 and 25;

select * from players_age_view;

create or replace view player_team_coach_view as
select p.player_name, p.player_nickname, t.team_name, c.coach_name, c.coach_nickname
from players p
join teams t on p.team_id = t.team_id
join coaches c on t.coach_id = c.coach_id;

select * from player_team_coach_view;

create or replace view player_team_view as
select p.player_name, p.player_nickname, p.player_role, p.nationality, t.team_name, t.founded_years
from players p
full outer join teams t on p.team_id = t.team_id;

select * from player_team_view order by player_name;

create or replace view team_sponsor_view as
select t.team_name, s.sponsor_name, s.industry
from teams t
left join teams_sponsors ts on t.team_id = ts.team_id
left join sponsors s on ts.sponsor_id = s.sponsor_id;

select * from team_sponsor_view;

create or replace view players_per_team as
select t.team_name, count(p.player_id) as number_of_players
from teams t
left join players p on t.team_id = p.team_id
group by t.team_name;

select * from players_per_team;

create or replace view avg_player_age_view as
select t.team_name, avg(p.age) as avg_player_age
from teams t
left join players p on t.team_id = p.team_id
group by t.team_name;

select * from avg_player_age_view order by avg_player_age desc;

create or replace view players_from_everywhere_not_russia as
select player_name, player_nickname, nationality
from players
minus 
select player_name, player_nickname, nationality
from players
where nationality = 'Russia';

select * from players_from_everywhere_not_russia order by nationality;

create or replace view player_younger_than_avg_view as
select player_name, player_nickname, age
from players
where age<(select avg(age) from players);

select * from player_younger_than_avg_view;

create or replace view teams_most_sponsors as
select t.team_name, count(ts.sponsor_id) as sponsor_count
from teams t
left join teams_sponsors ts on t.team_id = ts.team_id
group by t.team_name
having count(ts.sponsor_id) > (
    select avg(sponsor_count)
    from (
        select count(ts2.sponsor_id) as sponsor_count
        from teams t2
        left join teams_sponsors ts2 on t2.team_id = ts2.team_id
        group by t2.team_name
    )
);

select * from teams_most_sponsors;

create or replace trigger trg_sponsor_id
before insert on sponsors
for each row
begin
  :new.sponsor_id := sq_sponsor_id.nextval;
end;
/

CREATE SEQUENCE sq_sponsor_id START WITH 1 INCREMENT BY 1;

INSERT INTO sponsors (sponsor_name, industry, country)
VALUES ('New Sponsor', 'Software', 'USA');

CREATE OR REPLACE TRIGGER trg_check_player_age
BEFORE INSERT OR UPDATE ON players
FOR EACH ROW
BEGIN
  IF :NEW.age < 16 OR :NEW.age > 40 THEN
    RAISE_APPLICATION_ERROR(-20001, 'Player age must be between 16 and 40.');
  END IF;
END;
/

INSERT INTO players (player_id, player_name, player_nickname, age, player_role, nationality, team_id)
VALUES (sq_player_id.NEXTVAL, 'Too Young', 'Tiny', 14, 'Rifler', 'USA', 1);
