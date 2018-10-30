Title: Django ORM if you already know SQL
Date: 2018-10-29 23:57
Modified: 2018-10-29 23:57
Category:
Tags:
Slug: django-orm-for-sql-users
Summary: If you're migrating to Django from another MVC framework or an existing Django developer, chances are you already know SQL. In this post, I'll be illustrating how to use Django ORM by drawing analogies to equivalent SQL statements.
Status: draft

If you're migrating to Django from another MVC framework or an existing Django developer, chances are you already know SQL. 

In this post, I'll be illustrating how to use Django ORM by drawing analogies to equivalent SQL statements. Connecting new topic to your existing knowledge will help you learn to use the ORM faster.


Let us consider a simple base model for a person with attributes name, age and gender. 

![Person ER Diagram](https://i.imgur.com/t7Fs6oc.png)

To implement the above entity, we would model it as a table in SQL.

```sql
CREATE TABLE Person (
    id int,
    name varchar(50),
    age id int,
    gender varchar(10),
);
```

The same table is modelled in Django as a class which inherits from the base Model class. The ORM creates the equivalent table under the hood.

```python
class Person(models.Model):
    name = models.CharField(max_length=50, blank=True)
    age = models.IntegerField()
    gender = models.CharField(max_length=10, blank=True)
```

The most used data types are:  

|**SQL** | **Django**|
|--|--|
|`INT` | `IntegerField()`|
|`VARCHAR(n)` | `CharField(max_length=n)`|
|`TEXT` | `TextField()`|
|`FLOAT(n)` | `FloatField()`|
|`DATE` | `DateField()`|
|`TIME` | `TimeField()`|
|`DATETIME` | `DateTimeField()`|

The various queries we can use are:  
## SELECT Statement

**Fetch all rows**  
SQL:
```sql
SELECT *
FROM Person;
```

Django:
```python
Person.objects.all()
```

**Fetch specific columns**  
SQL:
```sql
SELECT name, age
FROM Person;
```

Django:
```python
Person.objects.only('name', 'age')
```

**Fetch distinct rows**  
SQL:
```sql
SELECT DISTINCT name, salary
FROM Person;
```

Django:
```python
Person.objects.distinct('name', 'salary')
```

**Fetch specific number of rows**  
SQL:
```sql
SELECT *
FROM Person
LIMIT 10;
```

Django:
```python
Person.objects.all()[:10]
```

**LIMIT AND OFFSET keywords**  
SQL:  
```sql
SELECT *
FROM Person
OFFSET 5
LIMIT 5;
```

Django:
```python
Person.objects.all()[5:10]
```

**Count number of rows**  
SQL:
```sql
SELECT count(*)
FROM Person;
```

Django:
```python
Person.objects.count()
```

## WHERE Clause

**Filter by single column**  
SQL:
```sql
SELECT *
FROM Person
WHERE id = 1;
```


Django:
```python
Person.objects.filter(id=1)
```

**Filter by comparison operators**  
SQL:
```sql
WHERE id > 1;
WHERE id >= 1;
WHERE id < 1;
WHERE id <= 1;
WHERE id != 1;
```


Django:
```python
Person.objects.filter(id__gt=1)
Person.objects.filter(id__gte=1)
Person.objects.filter(id__lt=1)
Person.objects.filter(id__lte=1)
Person.objects.exclude(id=1)
```

**BETWEEN Clause**  
SQL:
```sql
SELECT *
FROM Person 
WHERE age between 10 AND 20;
```

Django:
```python
Person.objects.filter(age__range=(10, 20))
```

**LIKE clause**  
SQL:
```sql
WHERE name like '%A%';
WHERE name like binary '%A%';
WHERE name like 'A%';
WHERE name like binary 'A%';
WHERE name like binary '%A';
```

Django:
```python
Person.objects.filter(name__icontains='A')
Person.objects.filter(name__contains='A')
Person.objects.filter(name__startswith='A')
Person.objects.filter(name__istartswith='A')
Person.objects.filter(name__endswith='A')
Person.objects.filter(name__iendswith='A')
```

**IN operator**  
SQL:
```sql
WHERE id in (1, 2);
```

Django:
```python
Person.objects.filter(id__in=[1, 2])
```

## AND, OR and NOT Operators  
SQL:
```sql
WHERE gender='male' AND salary > 20000;
```

Django:
```python
Person.objects.filter(gender='male', salary__gt=20000)
```

SQL:
```sql
WHERE gender='male' OR salary > 20000;
```

Django:
```python
from django.db.models import Q
Person.objects.filter(Q(gender='male') | Q(salary__gt=20000))
```

SQL:
```sql
WHERE NOT gender='male';
```

Django:
```python
Person.objects.exclude(gender='male')
```  


## NULL Values
SQL:
```sql
WHERE salary is NULL;
WHERE salary is NOT NULL;
```

Django:
```python
Person.objects.filter(salary__isnull=True)
Person.objects.filter(salary__isnull=False)
```

## ORDER BY Keyword  
**Ascending Order**  
SQL:
```sql
SELECT *
FROM Person
order by salary;
```

Django:
```python
Person.objects.order_by('salary')
```

**Descending Order**  
SQL:
```sql
SELECT *
FROM Person
ORDER BY salary DESC;
```

Django:
```python
Person.objects.order_by('-salary')
```

## INSERT INTO Statement
SQL:
```sql
INSERT INTO Person
VALUES ('Jack', '90000', '23');
```

Django:
```python
Person.objects.create(name='jack', salary=90000, age=23)
```

## UPDATE Statement
**Update single row**  
SQL:
```sql
UPDATE Person
SET salary = 20000
WHERE id = 1;
```

Django:
```python
person = Person.objects.get(id=1)
person.salary = 20000
person.save()
```

**Update multiple rows**  
SQL:
```sql
UPDATE Person
SET salary = salary * 1.5;
```

Django:
```python
from django.db.models import F

Person.objects.update(salary=F('salary')*1.5)
```

## DELETE Statement
**Delete all rows**  
SQL:
```sql
DELETE FROM Person;
```

Django:
```python
Person.objects.all().delete()
```

**Delete specific rows**  
SQL:
```sql
DELETE FROM Person
WHERE age < 10;
```

Django:
```python
Person.objects.filter(age__lt=10).delete()
```