SELECT 'ФИО: Корнев Игорь Сергеевич';
-- 1.1 запрос
SELECT *
FROM ratings
LIMIT 10;
-- 1.2 запрос
SELECT *
FROM links
WHERE moveid BETWEEN 100 and 1000
AND imdbid LIKE '%42'
LIMIT 10;
-- 2.1 запрос
SELECT imdbid
FROM links
INNER JOIN ratings USING(movieid)
WHERE rating = 5
LIMIT 10;
-- 3.1 запрос
SELECT COUNT(movieid)
FROM links
LEFT JOIN ratings USING(movieid)
WHERE ratings IS NULL;
-- 3.2 запрос
SELECT userid
FROM ratings
GROUP BY 1
HAVING AVG(rating) > 3.5
ORDER BY AVG(rating) DESC
LIMIT 10;
-- 4.1 запрос
SELECT imdbid
FROM (
	  SELECT imdbid
	  FROM links l
	  INNER JOIN ratings r USING(movieid)
	  GROUP BY 1
	  HAVING AVG(rating) > 3.5
	 ) AS t
LIMIT 10;
-- 4.2 запрос
WITH correct_ids AS (
SELECT userid
FROM ratings
GROUP BY 1
HAVING COUNT(userid) >= 10
)
SELECT AVG(rating)
FROM ratings r
INNER JOIN correct_ids USING(userid);
