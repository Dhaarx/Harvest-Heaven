import React, { useState, useEffect, useRef } from "react";
import { View, Text, StyleSheet, Image, ScrollView, Alert, Dimensions } from 'react-native';
import axios from "axios";
import Carousel from 'react-native-snap-carousel';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

const Prediction = ({ route }) => {
    const { base64String, imguri } = route.params;
    const [prediction, setPrediction] = useState(null);
    const [result, setResult] = useState('');
    const carouselRef = useRef(null);
    const [images, setImages] = useState([]);
    const [weatherData, setWeatherData] = useState([]);

    const sendToBackend = async () => {
        try {
            const response = await axios.post('http://172.16.123.164:5000/predict', {
                image: base64String,
            }, {
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                }
            });
            setPrediction(response.data.predicted_class);
            setResult(response.data.result);
            setImages(response.data.images);
        } catch (error) {
            console.error('Error uploading image:', error);
            Alert.alert('Error', 'Failed to upload image.');
        }
    };

    useEffect(() => {
        sendToBackend();
    }, []);

    useEffect(() => {
        if (images.length > 0) {
            const updatedWeatherData = images.map(im => ({ image: `data:image/png;base64,${im}` }));
            setWeatherData(updatedWeatherData);
        }
    }, [images]);

    const renderItem = ({ item }) => (
        <View style={styles.item}>
            <Image source={{ uri: item.image }} style={styles.weatherImage} />
        </View>
    );

    return (
        <ScrollView style={{ flex: 1 }}>
            <View style={{ flex: 1, backgroundColor: 'white', height: screenHeight }}>
                <View style={styles.container}>
                    {weatherData.length > 0 ? (
                        <Carousel
                            ref={carouselRef}
                            data={weatherData}
                            renderItem={renderItem}
                            sliderWidth={screenWidth}
                            itemWidth={screenWidth*0.91}
                            inactiveSlideScale={0.95}
                            inactiveSlideOpacity={0.7}
                            loop={false}
                        />
                    ) : (
                        <Text>Loading images...</Text>
                    )}
                </View>
                <View>
                    <Text>{result}</Text>
                </View>
            </View>
        </ScrollView>
    );
}

export default Prediction;

const styles = StyleSheet.create({
    image: {
        height: 200,
        width: '100%',
        marginBottom: 20,
        resizeMode: 'contain',
    },
    container: {
        marginTop: 20,
    },
    item: {
        borderRadius: 9,
    },
    weatherImage: {
        width:screenWidth*0.9,
        height: screenWidth*0.44,
    }
});
