import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import Home from './Home';
import Prediction from './Prediction';
import Macros from './Macros';
import Herbs from './Herbs';
import Diseases from './Diseases';
import Various from './Various';

const Homestack =createStackNavigator();

const Hometab=()=>{
    return(
        <Homestack.Navigator initialRouteName='Home'>
            <Homestack.Screen name="Home" component={Home}></Homestack.Screen>
            <Homestack.Screen name="Prediction" component={Prediction}></Homestack.Screen>
            <Homestack.Screen name="Macros" component={Macros}></Homestack.Screen>
            <Homestack.Screen name="Herbs" component={Herbs}></Homestack.Screen>
            <Homestack.Screen name="Diseases" component={Diseases}></Homestack.Screen>
            <Homestack.Screen name="Various" component={Various}></Homestack.Screen>
        </Homestack.Navigator>
    );
}

export default Hometab;

