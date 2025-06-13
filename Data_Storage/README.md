## 🗃️ Database Design for Movie Recommendation System

To efficiently manage user preferences and movie metadata for building a recommendation engine, we designed a **relational database schema** that is normalized and optimized for analytics and machine learning workflows.

### 🧩 Schema Overview

The database consists of the following tables:

#### 🎬 `movie`
| Column     | Type         | Description           |
|------------|--------------|-----------------------|
| movie_id   | INT (PK)     | Unique movie ID       |
| name       | VARCHAR      | Movie title           |
| year       | INT          | Release year          |

#### 🏷️ `genre`
| Column     | Type         | Description           |
|------------|--------------|-----------------------|
| genre_id   | INT (PK)     | Unique genre ID       |
| genre      | VARCHAR      | Genre name (e.g., Comedy) |

#### 🔗 `movie_genre`
| Column     | Type         | Description                     |
|------------|--------------|---------------------------------|
| movie_id   | INT (FK)     | References `movie.movie_id`     |
| genre_id   | INT (FK)     | References `genre.genre_id`     |

#### 👤 `users`
| Column         | Type         | Description                     |
|----------------|--------------|---------------------------------|
| user_id        | INT (PK)     | Unique user ID                  |
| gender         | CHAR         | Gender of the user              |
| age_id         | INT (FK)     | References `age.age_id`         |
| occupation_id  | INT (FK)     | References `occupation.occupation_id` |
| zipcode        | VARCHAR      | User's zip code                 |

#### 🕒 `rating`
| Column     | Type         | Description                     |
|------------|--------------|---------------------------------|
| user_id    | INT (FK)     | References `users.user_id`      |
| movie_id   | INT (FK)     | References `movie.movie_id`     |
| rating     | INT          | User rating (1–5)               |
| timestamp  | INT          | Unix timestamp                  |

#### 📅 `age`
| Column     | Type         | Description                     |
|------------|--------------|---------------------------------|
| age_id     | INT (PK)     | Unique age group ID             |
| age_range  | VARCHAR      | Age range (e.g., "18–24")       |

#### 💼 `occupation`
| Column         | Type         | Description              |
|----------------|--------------|--------------------------|
| occupation_id  | INT (PK)     | Unique occupation ID     |
| title          | VARCHAR      | Occupation name          |

---

### 🔗 Entity Relationships

- A **movie** can belong to multiple **genres**, and a **genre** can be linked to multiple **movies** → `movie_genre`
- A **user** can rate many **movies** → `rating`
- Each **user** is associated with an **age group** and an **occupation**

---

### 📊 ER Diagram

```mermaid
erDiagram
    movie ||--o{ movie_genre :
    genre ||--o{ movie_genre :
    users ||--o{ rating : gives
    movie ||--o{ rating : receives
    users }o--|| age : 
    users }o--|| occupation : "has"

