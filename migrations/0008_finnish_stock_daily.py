# Generated by Django 4.1.5 on 2023-01-27 21:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('marketBotApp', '0007_stock_1min_stock_30min_stock_5min_stock_60min_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='finnish_stock_daily',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('symbol', models.CharField(max_length=10)),
                ('open', models.FloatField()),
                ('high', models.FloatField()),
                ('low', models.FloatField()),
                ('close', models.FloatField()),
                ('volume', models.IntegerField()),
                ('bid', models.FloatField()),
                ('ask', models.FloatField()),
                ('turnover', models.FloatField()),
                ('trades', models.IntegerField()),
                ('date', models.DateField()),
            ],
        ),
    ]
