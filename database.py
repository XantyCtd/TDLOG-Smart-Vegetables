from peewee import*
db = SqliteDatabase('user.db')
#Définition de la bdd avec la classe user
class User(Model):
    username = CharField(unique=True)
    email = CharField()
    password = CharField()
    first_name = CharField()
    last_name = CharField()
    class Meta:
        database = db
    @property
    def is_active(self):
        return True
    @property
    def is_authenticated(self):
        return self.is_active
    @property
    def is_anonymous(self):
        return False
    def get_id(self):
        return self.id

db.create_tables([User])

