create table Coaches(
coach_id number(4,0) primary key,
coach_name varchar2(35) NOT NULL,
experience_years number(10,0)
);

create table Teams(
team_id number(4,0) primary key,
team_name varchar2(20) not null,
founded_years DATE,
country varchar2(20),
coach_id number(4,0),
tournament_id number(4,0),
constraint fk_coach foreign key (coach_id) references coaches(coach_id)
);

create table Players(
player_id number(4,0) primary key,
player_name varchar2(50) not null,
age number(4,0),
player_role varchar2(20),
nationality varchar2(25),
team_id number(4,0),
constraint fk_team foreign key (team_id) references teams(team_id)
);

create table Leagues(
league_id number(4,0) primary key,
league_name varchar2(35) not null,
season varchar2(20),
location varchar(100),
prize_pool decimal(12,2)
);

create table Sponsors(
sponsor_id number(4,0) primary key,
sponsor_name varchar2(35) not null,
industry varchar2(35)
);

create table Tournaments(
tournament_id number(4,0) primary key,
tournament_name varchar2(100) not null,
start_date Date,
end_date date,
prize_pool decimal(12,2),
league_id number(4,0),
constraint fk_league foreign key(league_id) references leagues(league_id)
);

CREATE TABLE Teams_Sponsors (
    team_id    NUMBER(4,0) PRIMARY KEY,
    sponsor_id NUMBER(4,0) PRIMARY KEY,

    CONSTRAINT fk_teams_sponsors_team
        FOREIGN KEY (team_id)
        REFERENCES Teams(team_id),
    CONSTRAINT fk_teams_sponsors_sponsor
        FOREIGN KEY (sponsor_id)
        REFERENCES Sponsors(sponsor_id)
);