# Generated by Django 4.1.5 on 2023-01-28 14:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('marketBotApp', '0009_finnish_stock_daily_average'),
    ]

    operations = [
        migrations.AlterField(
            model_name='adl',
            name='date',
            field=models.DateField(),
        ),
        migrations.AlterField(
            model_name='bbp8_model',
            name='date',
            field=models.DateField(),
        ),
        migrations.AlterField(
            model_name='obv',
            name='date',
            field=models.DateField(),
        ),
        migrations.AlterField(
            model_name='stock_daily',
            name='date',
            field=models.DateField(),
        ),
    ]