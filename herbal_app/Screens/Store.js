//Store.js

import React, { useEffect, useState } from 'react';
import { View, Text, FlatList,Image,TouchableOpacity, Linking  } from 'react-native';
import * as Location from 'expo-location';
import axios from 'axios';
const defaultImage = 'https://cdn1.vectorstock.com/i/1000x1000/51/85/veterinary-clinic-flat-vets-vector-26625185.jpg'; // Replace with your default image URL

const Store =()=>{
  const [location, setLocation] = useState(null);
  const [clinics, setClinics] = useState([]);
  const [errorMsg, setErrorMsg] = useState(null);

  useEffect(() => {
    (async () => {
      // Request location permission
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setErrorMsg('Permission to access location was denied');
        return;
      }

      // Get the current position of the user
      let location = await Location.getCurrentPositionAsync({});
      const { latitude, longitude } = location.coords;
      setLocation({ latitude, longitude });
      fetchNearbyClinics(latitude, longitude);
    })();
  }, []);

  const fetchNearbyClinics = async (latitude, longitude) => {
    const apiKey = 'AIzaSyDouSDXuZs-C61VHt6eJiIgP4ndfv41pDU';
    const radius = 20000; // Radius in meters
    const type = 'veterinary_care';
    const url = `https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=${latitude},${longitude}&radius=${radius}&type=${type}&key=${apiKey}`;
    try {
      const response = await axios.get(url);
      const clinicsWithPhotos = response.data.results.map((clinic) => {
        if (clinic.photos && clinic.photos.length > 0) {
          const photoReference = clinic.photos[0].photo_reference;
          clinic.photoUrl = `https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference=${photoReference}&key=${apiKey}`;
        }
        return clinic;
      });
      setClinics(clinicsWithPhotos);
    } catch (error) {
      console.error(error);
    }
  };
    const openGoogleMaps = (lat, lng) => {
      const url = `https://www.google.com/maps/search/?api=1&query=${lat},${lng}`;
      Linking.openURL(url);

  };

    return(
      <View>
      {errorMsg ? (
        <Text>{errorMsg}</Text>
      ) : clinics.length > 0 ? (
        <FlatList
        style={{backgroundColor:'#021526',borderRadius:20,borderColor:'#021526' }}
          data={clinics}
          keyExtractor={(item) => item.place_id}
          renderItem={({ item }) => (
            <View style={{ flexDirection: 'row',paddingLeft:10,paddingTop:10,paddingBottom:10,paddingRight:3, borderBottomWidth: 1, alignItems: 'center',backgroundColor:'white',borderRadius:20,marginBottom:8,borderColor:'white'}}>
              <Image
                source={{ uri: item.photoUrl || defaultImage }}
                style={{ width: 100, height: 100, marginRight: 10 }}
                resizeMode="cover"
              />
              <View style={{ flex: 1 }}>
                <TouchableOpacity onPress={() => openGoogleMaps(item.geometry.location.lat, item.geometry.location.lng)}>
                  <Text style={{ fontSize:22,fontWeight: 'bold', color: '#1E2A5E' }}>
                    {item.name}
                  </Text>
                </TouchableOpacity>
                <Text style={{ fontSize:16,fontWeight: 'bold', color: '#405D72' }}>{item.vicinity}</Text>
              </View>
            </View>
          )}
        />
      ) : (
        <Text>Loading...</Text>
      )}
    </View>
  );
}

export default Store;
