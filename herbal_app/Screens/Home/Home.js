import React, { useEffect, useRef, useState } from 'react';
import { View , Text, StyleSheet, Dimensions ,TouchableOpacity ,Image,Alert,ScrollView } from 'react-native';
import Carousel from 'react-native-snap-carousel'
import * as ImagePicker from 'expo-image-picker';
import { CommonActions } from '@react-navigation/native';
import * as Location from 'expo-location';

const { width: screenWidth ,height:screenHeight} = Dimensions.get('window');

// routing
const Macroroute=(navigation)=>{
  navigation.dispatch(
    CommonActions.navigate({
      name:'Macros',
      })
  );
};
const Diseasesroute=(navigation)=>{
  navigation.dispatch(
    CommonActions.navigate({
      name:'Diseases',
      })
  );
};
const Herbsroute=(navigation)=>{
  navigation.dispatch(
    CommonActions.navigate({
      name:'Herbs',
      })
  );
};
const Variousroute=(navigation)=>{
  navigation.dispatch(
    CommonActions.navigate({
      name:'Various',
      })
  );
};


const Home =({navigation})=>{

  const carouselRef = useRef(null);
  const [permission,setpermission] = useState(null);
  const [location, setLocation] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [Weather, setWeather] = useState(null);

 

    useEffect(()=>{
      (async() =>{
        const camerastatus = await ImagePicker.requestCameraPermissionsAsync();
        setpermission(camerastatus.status === 'granted');

        let { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== 'granted') {
            setErrorMsg('Permission to access the location was denied');
            return;
        }

        let locations = await Location.getCurrentPositionAsync({});
        setLocation(locations);
      })();
    },[]);

    useEffect(() => {
      if (location) {
        (async () => {
          try {
            const response = await fetch(
              `https://api.openweathermap.org/data/2.5/weather?lat=${location.coords.latitude}&lon=${location.coords.longitude}&appid=a2eda770b7bb5c185700e16a1c750459&units=metric`
            );
            const weatherData = await response.json();
            setWeather(weatherData);
          } catch (error) {
            setErrorMsg(`Error fetching weather data: ${error.message}`);
          }
        })();
      }
    }, [location]);
  
    if (errorMsg) {
    } else if (location) {
    }    

    const captureimg=async()=>{
      let result = await ImagePicker.launchCameraAsync({
        allowsEditing :true,
        aspect:[4,3],
        quality:1,
        base64:true,
      });
      
      if (!result.canceled && result.assets) {
        const base64String = result.assets[0].base64;
        const imguri = result.assets[0].uri; 

        if(base64String)
        {
          console.log('found the base64'); 
        }
        else{
          console.log('No base64');
        }     
        
        navigation.dispatch(
          CommonActions.navigate({
            name:'Prediction',
            params:{
              base64String:base64String,
              imguri:imguri
            }
            })
        );
        console.log("Page has been redirected to : Prediction");
      }
      else {
        console.log("No image selected or no base64 value found.");
      }  
      
    };

    const weatherData =Weather? [
      { area:`${Weather.name}`, weather:`${Weather.weather[0].description}`, temp:`${Weather.main.temp.toFixed(2)}`, image:`https://openweathermap.org/img/wn/${Weather.weather[0].icon}@2x.png` },
      { area:'Chettipalayam' ,weather: 'sunny', temp: '27.34',image: `https://openweathermap.org/img/wn/${Weather.weather[0].icon}@2x.png`, },
    ]:[];

    const renderItem = ({ item }) => {
      return (
        <View style={styles.item}>
          <View>
            <Text style={styles.areaText}>{item.area}</Text>
            <Text style={styles.weatherText}>{item.weather}</Text>
          </View>
          <View style={styles.rightContainer}>
            <Text style={styles.tempText}>{item.temp}°C</Text>
            <Image source={{ uri: item.image }} style={styles.weatherImage} />
          </View>
        </View>
      );
    };
    
    return(
      <ScrollView style={{flex:1}}>
        <View style={{backgroundColor:'white', flex:1, height:screenHeight}}>       
          
          {/* it is the carousel */}
          <View style={styles.container}>
            <Carousel
              ref={carouselRef}
              data={weatherData}
              renderItem={renderItem}
              sliderWidth={screenWidth}
              itemWidth={screenWidth * 0.89}
              inactiveSlideScale={0.95}
              inactiveSlideOpacity={0.7}
              loop={false}
            />
          </View>
          
          {/* it is the topic */}
          <View style={styles.topic}>
            <Text style={styles.topictext}>Heal your disease</Text>
          </View>

          {/* it is the camera view */}
          <View style={styles.camera}>
            <Image source={require('../icons/tea.png')} style={styles.icon1}></Image>
            <Image source={require('../icons/login.png')} style={styles.icon2}></Image>
            <Image source={require('../icons/herbal (1).png')} style={styles.icon3}></Image>
            <Image source={require('../icons/chevron.png')} style={styles.icon4}></Image>
            <Image source={require('../icons/chevron.png')} style={styles.icon5}></Image>
            <Text style={styles.cameratxt1}>Take a {'\n'} picture</Text>
            <Text style={styles.cameratxt2}>see{'\n'}diagnosis</Text>
            <Text style={styles.cameratxt3}>Get{'\n'}Medicines</Text>
            <TouchableOpacity style={styles.picturebtn} onPress={captureimg}>
              <Text style={styles.picturebtntxt}>Take a picture</Text>
            </TouchableOpacity>             
          </View>  

          {/* it is tiles */}
          <TouchableOpacity style={styles.tile1} onPress={() => Macroroute(navigation)} >
            <Image source={require('../icons/ferti.jpg')} style={styles.tileimg1}></Image>           
            <Text style={styles.tiletxt1}>Fertilizer{'\n'}calculator</Text>
            <Image source={require('../icons/chevron.png')} style={styles.tileicon1}></Image>          
          </TouchableOpacity>
          <TouchableOpacity style={styles.tile2} onPress={()=>Diseasesroute(navigation)}> 
            <Image source={require('../icons/prevention.png')} style={styles.tileimg2}></Image>          
            <Text style={styles.tiletxt2}>Diseases &{'\n'}Diagnosis</Text>
            <Image source={require('../icons/chevron.png')} style={styles.tileicon2}></Image>
          </TouchableOpacity>              
          <TouchableOpacity style={styles.tile3} onPress={()=>Herbsroute(navigation)}>    
            <Image source={require('../icons/herb.png')} style={styles.tileimg3}></Image>       
            <Text style={styles.tiletxt3}>Crop{'\n'}Recommedation</Text>
            <Image source={require('../icons/chevron.png')} style={styles.tileicon3}></Image>
          </TouchableOpacity>            
          <TouchableOpacity style={styles.tile4} onPress={()=>Variousroute(navigation)}> 
            <Image source={require('../icons/vet_doc.jpg')} style={styles.tileimg4}></Image>                    
            <Text style={styles.tiletxt4}>Veterinarian{'\n'}Connect</Text>
            <Image source={require('../icons/chevron.png')} style={styles.tileicon4}></Image>
          </TouchableOpacity>  
        
        </View>
      </ScrollView>
    );
};


const styles = StyleSheet.create({
    container: {
      marginTop: 20,
    },
    item: {
      backgroundColor: '#F7F7F7',
      borderColor:'#0157DB',
      borderWidth:2,
      paddingTop:20,
      borderRadius: 50,
      height: 80,
      padding: 20,
      flexDirection: 'row',
      alignItems: 'center', 
      justifyContent: 'space-between',

    },
    areaText:{
      fontSize:18,
      fontWeight:'bold'
    },
    weatherText: {
      fontSize: 15,
    },
    tempText: {
      fontSize: 16,
      color: '#888',
      marginRight:5,
    },
    weatherImage: {
      width: 70,
      height: 70,
      marginLeft:0,
    },
    rightContainer: {
      flexDirection: 'row',
      alignItems: 'center', 
    },
    camera: {
      backgroundColor: '#F7F7F7',
      borderRadius:10,
      elevation:4,
      top:50,
      left:20,
      width:318,
      height:220,
      position:'relative',
    },
    topic:{
      top:35,
      left:20,
      color:'black',
    },
    topictext:{
      fontSize:18
    },
    picturebtn:{
      position:'absolute',
      top:160,
      width:260,
      left:27,
      height:35,
      borderRadius:80,
      backgroundColor:'#0157DB',
      alignItems:'center',
      justifyContent:'center'
    },
    picturebtntxt:{
      fontSize:16.5,
      color:'white'
    },
    icon1:{
      position:'absolute',
      top:18,
      left:19.5,
      height:60,
      width:60
    },
    icon2:{
      position:'absolute',
      top:18,
      left:125,
      height:55,
      width:55
    },
    icon3:
    {    
      position:'absolute',
      top:18,
      left:230,
      height:60,
      width:60      
    },
    icon4:{
      position:'absolute',
      top:42,
      left:89,
      height:25,
      width:30  
    },
    icon5:{
      position:'absolute',
      top:42,
      left:189,
      height:25,
      width:30  
    },
    cameratxt1:{
      position:'absolute',
      top:89,
      left:31.5,
      textAlign:'center'
    },
    cameratxt2:{
      position:'absolute',
      top:89,
      left:132.5,
      textAlign:'center'
    },
    cameratxt3:{
      position:'absolute',
      top:89,
      left:232.5,
      textAlign:'center'
    },
    tile1:{
      top:65,
      height:116,
      width:153,
      backgroundColor:'#E7F0FF',
      position:'relative',
      left:20,
      borderRadius:10,
      elevation:4
    },
    tile2:{
      top:77,
      height:116,
      width:153,
      backgroundColor:'#E7F0FF',
      position:'relative',
      left:20,
      borderRadius:10,
      elevation:4
    },
    tile3:{
      top:408.2,
      height:116,
      width:153,
      backgroundColor:'#E7F0FF',
      position:'absolute',
      left:184.2,
      borderRadius:10,
      elevation:4
    },
    tile4:{
      top:536.5,
      height:116,
      width:153,
      backgroundColor:'#E7F0FF',
      position:'absolute',
      left:184.2,
      borderRadius:10,
      elevation:4
    },
    tileicon1:{
      position:'absolute',
      top:42,
      left:113,
      height:25,
      width:30,
    },
    tileicon2:{
      position:'absolute',
      top:42,
      left:113,
      height:25,
      width:30 
    },
    tileicon3:{
      position:'absolute',
      top:42,
      left:113,
      height:25,
      width:30 
    },
    tileicon4:{
      position:'absolute',
      top:42,
      left:113,
      height:25,
      width:30 
    },
    tiletxt1:{
      top:60,
      fontSize:17,
      left:13,
    },
    tiletxt2:{
      top:60,
      fontSize:17,
      left:13
    },
    tiletxt3:{
      top:60,
      fontSize:17,
      left:13
    },
    tiletxt4:{
      top:60,
      fontSize:17,
      left:13
    },
    tileimg1:{
      position:'absolute',
      height:35,
      width:35,
      left:13,
      top:10,
      backgroundColor: '#F5F7FE',
      borderRadius: 14,
    },
    tileimg2:{
      position:'absolute',
      height:35,
      width:35,
      left:13,
      top:10,
      backgroundColor: '#F5F7FE',
      borderRadius: 14,
    },
    tileimg3:{
      position:'absolute',
      height:35,
      width:35,
      left:13,
      top:10,
      backgroundColor: '#F5F7FE',
      borderRadius: 14,
    },
    tileimg4:{
      position:'absolute',
      height:35,
      width:35,
      left:13,
      top:10,
      backgroundColor: '#F5F7FE',
      borderRadius: 14,
    }
  });

export default Home;