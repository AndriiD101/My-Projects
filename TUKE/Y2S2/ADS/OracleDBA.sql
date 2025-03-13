create table Coaches(
coach_id number(4,0) primary key,
coach_name varchar2(35) NOT NULL,
coach_nickname varchar2(20) NOT NULL,
experience_years number(10,0)
);

create sequence sq_coach_id
start with 1
increment by 1
minvalue 1
maxvalue 1000
NOCYCLE;

INSERT INTO coaches (coach_id, coach_name, coach_nickname, experience_years) 
values(sq_coach_id.nextval, 'Andrij Ghorodensjkyj', 'B1ad3', 6);
INSERT INTO coaches (coach_id, coach_name, coach_nickname, experience_years) 
values(sq_coach_id.nextval, 'Sergey Shavayev', 'hally', 5);
INSERT INTO coaches (coach_id, coach_name, coach_nickname, experience_years) 
values(sq_coach_id.nextval, 'Wiktor Wojtas', 'TaZ', 3);
INSERT INTO coaches (coach_id, coach_name, coach_nickname, experience_years) 
values(sq_coach_id.nextval, 'Erdenedalai Bayanbat', 'maaRaa', 9);
INSERT INTO coaches (coach_id, coach_name, coach_nickname, experience_years) 
values(sq_coach_id.nextval, 'Rémy Quoniam', 'XTQZZZ', 11);
INSERT INTO coaches (coach_id, coach_name, coach_nickname, experience_years) 
values(sq_coach_id.nextval, 'Dennis Nielsen', 'sycrone', 4);
INSERT INTO coaches (coach_id, coach_name, coach_nickname, experience_years) 
values(sq_coach_id.nextval, 'Casper Due', 'ruggah', 9);
INSERT INTO coaches (coach_id, coach_name, coach_nickname, experience_years) 
values(sq_coach_id.nextval, 'Eetu Saha', 'sAw', 6);
INSERT INTO coaches (coach_id, coach_name, coach_nickname, experience_years) 
values(sq_coach_id.nextval, 'Filip Kubski', 'NEO', 2);
INSERT INTO coaches (coach_id, coach_name, coach_nickname, experience_years) 
values(sq_coach_id.nextval, 'Sezgin Kalaycis', 'Fabre', 3);

select * from coaches;

create table Leagues(
league_id number(4,0) primary key,
league_name varchar2(35) not null,
season varchar2(20),
location varchar(100),
prize_pool decimal(12,2)
);

INSERT INTO Leagues (league_id, league_name, season, location, prize_pool) 
VALUES (1, 'ESL Pro League', 'Season 20', 'Global', 850000.00);
INSERT INTO Leagues (league_id, league_name, season, location, prize_pool) 
VALUES (2, 'Intel Extreme Masters', 'Season 18', 'Katowice, Poland', 1000000.00);
INSERT INTO Leagues (league_id, league_name, season, location, prize_pool) 
VALUES (3, 'BLAST Premier', 'Spring 2024', 'Copenhagen, Denmark', 425000.00);
INSERT INTO Leagues (league_id, league_name, season, location, prize_pool) 
VALUES (4, 'DreamHack Masters', 'Winter 2024', 'Jönköping, Sweden', 250000.00);
INSERT INTO Leagues (league_id, league_name, season, location, prize_pool) 
VALUES (5, 'FACEIT Pro League', 'Season 10', 'Online', 50000.00);
INSERT INTO Leagues (league_id, league_name, season, location, prize_pool) 
VALUES (6, 'ESEA Premier', 'Season 40', 'North America', 200000.00);
INSERT INTO Leagues (league_id, league_name, season, location, prize_pool) 
VALUES (7, 'EPICENTER', '2024', 'Moscow, Russia', 500000.00);
INSERT INTO Leagues (league_id, league_name, season, location, prize_pool) 
VALUES (8, 'ELEAGUE', 'Season 5', 'Atlanta, USA', 150000.00);
INSERT INTO Leagues (league_id, league_name, season, location, prize_pool) 
VALUES (9, 'StarLadder i-League', 'Season 8', 'Shanghai, China', 300000.00);
INSERT INTO Leagues (league_id, league_name, season, location, prize_pool) 
VALUES (10, 'PGL Esports', '2024', 'Bucharest, Romania', 250000.00);

select * from leagues;

create table Tournaments(
tournament_id number(4,0) primary key,
tournament_name varchar2(100) not null,
start_date Date,
end_date date,
prize_pool decimal(12,2),
league_id number(4,0),
constraint fk_league foreign key(league_id) references leagues(league_id)
);

INSERT INTO Tournaments (tournament_id, tournament_name, start_date, end_date, prize_pool, league_id) 
VALUES (1, 'ESL Pro League Season 20', TO_DATE('2024-09-03', 'YYYY-MM-DD'), TO_DATE('2024-09-22', 'YYYY-MM-DD'), 750000.00, 1);
INSERT INTO Tournaments (tournament_id, tournament_name, start_date, end_date, prize_pool, league_id) 
VALUES (2, 'Intel Extreme Masters Katowice 2024', TO_DATE('2024-01-31', 'YYYY-MM-DD'), TO_DATE('2024-02-11', 'YYYY-MM-DD'), 1000000.00, 2);
INSERT INTO Tournaments (tournament_id, tournament_name, start_date, end_date, prize_pool, league_id) 
VALUES (3, 'BLAST Premier Spring Final 2024', TO_DATE('2024-06-14', 'YYYY-MM-DD'), TO_DATE('2024-06-20', 'YYYY-MM-DD'), 425000.00, 3);
INSERT INTO Tournaments (tournament_id, tournament_name, start_date, end_date, prize_pool, league_id) 
VALUES (4, 'DreamHack Masters Winter 2024', TO_DATE('2024-11-28', 'YYYY-MM-DD'), TO_DATE('2024-12-06', 'YYYY-MM-DD'), 250000.00, 4);
INSERT INTO Tournaments (tournament_id, tournament_name, start_date, end_date, prize_pool, league_id) 
VALUES (5, 'FACEIT Global Summit: PUBG Classic', TO_DATE('2024-04-16', 'YYYY-MM-DD'), TO_DATE('2024-04-21', 'YYYY-MM-DD'), 400000.00, 5);
INSERT INTO Tournaments (tournament_id, tournament_name, start_date, end_date, prize_pool, league_id) 
VALUES (6, 'ESEA Season 40 Global Challenge', TO_DATE('2024-05-10', 'YYYY-MM-DD'), TO_DATE('2024-05-12', 'YYYY-MM-DD'), 75000.00, 6);
INSERT INTO Tournaments (tournament_id, tournament_name, start_date, end_date, prize_pool, league_id) 
VALUES (7, 'EPICENTER 2024', TO_DATE('2024-12-17', 'YYYY-MM-DD'), TO_DATE('2024-12-22', 'YYYY-MM-DD'), 500000.00, 7);
INSERT INTO Tournaments (tournament_id, tournament_name, start_date, end_date, prize_pool, league_id) 
VALUES (8, 'ELEAGUE CS:GO Invitational 2024', TO_DATE('2024-01-25', 'YYYY-MM-DD'), TO_DATE('2024-01-27', 'YYYY-MM-DD'), 150000.00, 8);
INSERT INTO Tournaments (tournament_id, tournament_name, start_date, end_date, prize_pool, league_id) 
VALUES (9, 'StarLadder i-League StarSeries Season 8', TO_DATE('2024-10-21', 'YYYY-MM-DD'), TO_DATE('2024-10-27', 'YYYY-MM-DD'), 500000.00, 9);
INSERT INTO Tournaments (tournament_id, tournament_name, start_date, end_date, prize_pool, league_id) 
VALUES (10, 'PGL Major Stockholm 2024', TO_DATE('2024-10-26', 'YYYY-MM-DD'), TO_DATE('2024-11-07', 'YYYY-MM-DD'), 2000000.00, 10);

select * from tournaments;

create table Teams(
team_id number(4,0) primary key,
team_name varchar2(20) not null,
founded_years DATE,
country varchar2(20),
coach_id number(4,0),
constraint fk_coach foreign key (coach_id) references coaches(coach_id)
);

INSERT INTO Teams (team_id, team_name, founded_years, country, coach_id) 
VALUES(1, 'NaVi', TO_DATE('2009-12-17', 'YYYY-MM-DD'), 'Ukraine', 1);

INSERT INTO Teams (team_id, team_name, founded_years, country, coach_id) 
VALUES(2, 'Team Spirit', TO_DATE('2015-06-15', 'YYYY-MM-DD'), 'Russia', 2);

INSERT INTO Teams (team_id, team_name, founded_years, country, coach_id) 
VALUES(3, 'G2 Esports', TO_DATE('2013-01-01', 'YYYY-MM-DD'), 'Germany', 3);

INSERT INTO Teams (team_id, team_name, founded_years, country, coach_id) 
VALUES(4, 'TheMongolz', TO_DATE('2016-07-01', 'YYYY-MM-DD'), 'Mongolia', 4);

INSERT INTO Teams (team_id, team_name, founded_years, country, coach_id) 
VALUES(5, 'Team Vitality', TO_DATE('2013-05-01', 'YYYY-MM-DD'), 'France', 5);

INSERT INTO Teams (team_id, team_name, founded_years, country, coach_id) 
VALUES(6, 'MOUZ', TO_DATE('2016-09-01', 'YYYY-MM-DD'), 'Germany', 6);

INSERT INTO Teams (team_id, team_name, founded_years, country, coach_id) 
VALUES(7, 'Astralis', TO_DATE('2016-07-01', 'YYYY-MM-DD'), 'Denmark', 7);

INSERT INTO Teams (team_id, team_name, founded_years, country, coach_id) 
VALUES(8, 'Heroic', TO_DATE('2016-01-01', 'YYYY-MM-DD'), 'Denmark', 8);

INSERT INTO Teams (team_id, team_name, founded_years, country, coach_id) 
VALUES(9, 'FaZe Clan', TO_DATE('2010-05-30', 'YYYY-MM-DD'), 'USA', 9);

INSERT INTO Teams (team_id, team_name, founded_years, country, coach_id) 
VALUES(10, 'Eternal Fire', TO_DATE('2021-10-21', 'YYYY-MM-DD'), 'Turkey', 10);

select * from teams;

create table Players(
player_id number(4,0) primary key,
player_name varchar2(50) not null,
player_nickname varchar2(25) not null,
age number(4,0),
player_role varchar2(20),
nationality varchar2(25),
team_id number(4,0),
constraint fk_team foreign key (team_id) references teams(team_id)
);

select * from players;

create sequence sq_player_id
start with 1
increment by 1
minvalue 1
maxvalue 1000
NOCYCLE;

--NaVi
INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval, 'Aleksi Virolainen', 'Aleksib', 26, 'In-Game Leader', 'Finland', 1);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval, 'Mihai Ivan', 'iM', 23, 'Rifler', 'Romania', 1);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval, 'Justinas Lekavicius', 'jL', 24, 'Rifler', 'Lithuania', 1);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval, 'Igor Zhdanov', 'w0nderful', 18, 'AWPer', 'Ukraine', 1);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval, 'Valerii Vakhovskyi', 'b1t', 20, 'Rifler', 'Ukraine', 1);

--Team Spirit
INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval, 'Leonid Vishnyakov', 'chopper', 25, 'In-Game Leader', 'Russia', 2);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval, 'Dmitriy Sokolov', 'sh1ro', 22, 'AWPer', 'Russia', 2);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval, 'Boris Vorobiev', 'magixx', 20, 'Rifler', 'Russia', 2);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval, 'Aleksei Zimin', 'Zont1x', 19, 'Rifler', 'Ukraine', 2);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval, 'Aleksandr Fomin', 'donk', 16, 'Rifler', 'Russia', 2);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Rasmus Nielsen','HooXi',28,'In-Game Leader','Denmark',3);

--G2
INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Nikola Kovač','NiKo',26,'Rifler','Bosnia and Herzegovina',3);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Nemanja Kovač','huNter-',27,'Rifler','Bosnia and Herzegovina',3);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Ilya Osipov','m0NESY',18,'AWPer','Russia',3);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Justin Savage','jks',27,'Rifler','Australia',3);

--mongolz
INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval, 'Munkhbold Baatar', 'Senzu', 22, 'AWPer', 'Mongolia', 4);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval, 'Tsogookh Tserendorj', 'Tsogookh', 24, 'Rifler', 'Mongolia', 4);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval, 'Garidmagnai Byambasuren', 'bLitz', 25, 'In-Game Leader', 'Mongolia', 4);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval, 'Nergui Timo', 'nTimo', 23, 'Rifler', 'Mongolia', 4);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval, 'Ganzorig Erdene', '910', 24, 'Support', 'Mongolia', 4);

--vitality
INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id)
VALUES (sq_player_id.nextval,'Dan Madesclaire','apEX',30,'In-Game Leader','France',5);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Peter Rasmussen','dupreeh',30,'Rifler','Denmark',5);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Mathieu Herbaut','ZywOo',23,'AWPer','France',5);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Shahar Shushan','flameZ',20,'Rifler','Israel',5);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Nabil Bensalem','NaVi',22,'Support','France',5);

--mouz
INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval,'Benjamin Kanter','BeKan',24,'In-Game Leader','Germany',6);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval,'Kamil Piotrowski','Spiin',22,'AWPer','Poland',6);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval,'Ádám Torzsás','Torzs',20,'Rifler','Hungary',6);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval,'Jimi Salo','Jimpphat',18,'Rifler','Finland',6);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval,'Dorian Berman','xertioN',19,'Rifler','Israel',6);

--astalis
INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Nicolai Reedtz','dev1ce',28,'AWPer','Denmark',7);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Lukas Rossander','gla1ve',28,'In-Game Leader','Denmark',7);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Martin Lund','stavn',21,'Rifler','Denmark',7);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Jacob Nygaard','jabbi',20,'Rifler','Denmark',7);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Johannes Nielsen','Staehr',18,'Rifler','Denmark',7);

--heroic
INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Alvaro Garcia','SunPayus',24,'AWPer','Spain',8);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Rasmus Nielsen','HooXi',28,'In-Game Leader','Denmark',8);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Timur Rasulov','TMR',23,'Rifler','Kazakhstan',8);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Li Ze','L1Z',22,'Support','China',8);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Özgür Eker','woxicSV',24,'AWPer','Turkey',8);

--faze clan
INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval,'Håvard Nygaard','rain',29,'Rifler','Norway',9);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval,'Finn Andersen','karrigan',33,'In-Game Leader','Denmark',9);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval,'Emil Dahlgren','EDGE',24,'Rifler','Sweden',9);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval,'Helvijs Saukants','broky',21,'AWPer','Latvia',9);

INSERT INTO Players (player_id, player_name, player_nickname, age, player_role, nationality, team_id) 
VALUES (sq_player_id.nextval,'Kai Finnex','Finnex',26,'Support','USA',9);

--EF
INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Ismailcan Dortkardes','XANTARES',27,'Rifler','Turkey',10);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Engin Küpeli','MAJ3R',31,'In-Game Leader','Turkey',10);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Özgür Eker','w0xic',24,'AWPer','Turkey',10);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Xicadian Mehmet','Xicadian',23,'Support','Turkey',10);

INSERT INTO Players (player_id,player_name,player_nickname,age,player_role,nationality,team_id) 
VALUES (sq_player_id.nextval,'Emre Jobaa','jobAAA',22,'Rifler','Turkey',10);

create table Sponsors(
sponsor_id number(4,0) primary key,
sponsor_name varchar2(35) not null,
industry varchar2(35),
country varchar2(50)
);

-- Inserting sponsors
INSERT INTO Sponsors (sponsor_id, sponsor_name, industry, country)
VALUES (1, '1xBet', 'Gambling', 'Russia');

INSERT INTO Sponsors (sponsor_id, sponsor_name, industry, country)
VALUES (2, 'Red Bull', 'Beverages', 'Austria');

INSERT INTO Sponsors (sponsor_id, sponsor_name, industry, country)
VALUES (3, 'Logitech', 'Peripherals', 'Switzerland');

INSERT INTO Sponsors (sponsor_id, sponsor_name, industry, country)
VALUES (4, 'Kaspersky', 'Cybersecurity', 'Russia');

INSERT INTO Sponsors (sponsor_id, sponsor_name, industry, country)
VALUES (5, 'Tezos', 'Blockchain', 'Switzerland');

INSERT INTO Sponsors (sponsor_id, sponsor_name, industry, country)
VALUES (6, 'Hummel', 'Sportswear', 'Denmark');

INSERT INTO Sponsors (sponsor_id, sponsor_name, industry, country)
VALUES (7, 'ASUS ROG', 'Computer Hardware', 'Taiwan');

INSERT INTO Sponsors (sponsor_id, sponsor_name, industry, country)
VALUES (8, 'Pringles', 'Food and Beverage', 'USA');

INSERT INTO Sponsors (sponsor_id, sponsor_name, industry, country)
VALUES (9, 'Mastercard', 'Financial Services', 'USA');

INSERT INTO Sponsors (sponsor_id, sponsor_name, industry, country)
VALUES (10, 'Lenovo', 'Computer Hardware', 'China');

select * from sponsors;
select * from teams;

CREATE TABLE Teams_Sponsors (
    team_id    NUMBER(4,0),
    sponsor_id NUMBER(4,0),

    -- Composite primary key
    PRIMARY KEY (team_id, sponsor_id),

    -- Foreign Keys
    CONSTRAINT fk_teams_sponsors_team
        FOREIGN KEY (team_id)
        REFERENCES Teams(team_id),
    CONSTRAINT fk_teams_sponsors_sponsor
        FOREIGN KEY (sponsor_id)
        REFERENCES Sponsors(sponsor_id)
);

INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (1, 1);  
INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (1, 2);  

INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (2, 3); 
INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (2, 4); 

INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (3, 5); 
INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (3, 6);  

INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (4, 1);
INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (4, 7); 

INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (5, 2);  
INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (5, 8);

INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (6, 9);  
INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (6, 10); 

INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (7, 3); 
INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (7, 5);  

INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (8, 6); 
INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (8, 7);  

INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (9, 2);  
INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (9, 4);  

INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (10, 8); 
INSERT INTO Teams_Sponsors (team_id, sponsor_id) VALUES (10, 10);

select * from teams_sponsors;

commit;