const express = require('express');
const cors = require('cors');
const app = express();
const connection = require('./config/connection');
const logger = require('morgan');
const axios = require('axios');
app.use(cors());
app.use(express.json());
app.use(logger("dev"))

const session=require("express-session");

app.post('/nodepredict', (req, res)=>{
  console.log(req.body.predict)
  axios.post('http://localhost:5000/SmartClassroom', {predict: req.body.predict}).then(data=>{
    res.send(data.data)
    console.log(data)
  }).catch(err=>{
    console.log(err)
  })
})

app.post('/nodeNewspaperShelves', (req, res)=>{
  console.log(req.body.predict)
  axios.post('http://localhost:5000/NewspaperShelvesPredict', {predict: req.body.predict}).then(data=>{
    res.send(data.data)
    console.log(data)
  }).catch(err=>{
    console.log(err)
  })
})

app.post('/nodeSortingHat', (req, res)=>{
  console.log(req.body.predict)
  axios.post('http://localhost:5000/SortingHatPredict', {predict: req.body.predict}).then(data=>{
    res.send(data.data)
    console.log(data)
  }).catch(err=>{
    console.log(err)
  })
})

app.post('/nodeTouristinInfo', (req, res)=>{
  console.log(req.body.predict)
  axios.post('http://localhost:5000/TouristinInfoPredict', {predict: req.body.predict}).then(data=>{
    res.send(data.data)
    console.log(data)
  }).catch(err=>{
    console.log(err)
  })
})

app.post('/nodeMakeMeHappy', (req, res)=>{
  console.log(req.body.predict)
  axios.post('http://localhost:5000/MakeMeHappyPredict', {predict: req.body.predict}).then(data=>{
    res.send(data.data)
    console.log(data)
  }).catch(err=>{
    console.log(err)
  })
})

app.listen(4500, () => { console.log("server is up and runing") })