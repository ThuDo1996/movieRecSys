USE movieRecSys

### create movie table
CREATE TABLE movies(
movie_id INT PRIMARY KEY,
name VARCHAR(100),
year INT
);

### create genre table
CREATE TABLE genre(
genre_id INT PRIMARY KEY,
genre VARCHAR(100) NOT NULL
);

### create movie-genre table
CREATE TABLE movie_genre(
movie_id INT,
genre_id INT,
FOREIGN KEY (movie_id) REFERENCES movies(movie_id),
FOREIGN KEY (genre_id) REFERENCES genre(genre_id)
);

### create table age
CREATE TABLE age(
age_id INT PRIMARY KEY,
age_range VARCHAR(100) NOT NULL
);

### create table occupation
CREATE TABLE occupation(
occupation_id INT PRIMARY KEY,
title VARCHAR(100) NOT NULL
);

### create table users
CREATE TABLE users(
user_id INT PRIMARY KEY,
gender VARCHAR(100) NOT NULL,
age_id INT,
occupation_id INT,
zipcode VARCHAR(100),
FOREIGN KEY (age_id) REFERENCES age(age_id),
FOREIGN KEY (occupation_id) REFERENCES occupation(occupation_id)
);

#### create table ratings
CREATE TABLE ratings(
user_id INT,
movie_id INT,
rating INT NOT NULL,
timestamp INT,
FOREIGN KEY (user_id) REFERENCES users(user_id),
FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);

