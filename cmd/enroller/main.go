package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	authenticator "github.com/Guillaume-Boutry/face-authenticator-wrapper"
	"github.com/Guillaume-Boutry/grpc-backend/pkg/face_authenticator"
	"github.com/golang/protobuf/proto"
	"log"
	"os"
	"path/filepath"
	"time"

	cloudevents "github.com/cloudevents/sdk-go/v2"
	"github.com/kelseyhightower/envconfig"
)

// Alias to dlib type
type FeatureMatrix authenticator.Dlib_matrix_Sl_float_Sc_0_Sc_1_Sg_

type Receiver struct {
	client cloudevents.Client

	// If the K_SINK environment variable is set, then events are sent there,
	// otherwise we simply reply to the inbound request.
	Target string `envconfig:"K_SINK"`
	// Channel to send work
	jobChannel chan *work
}

func main() {
	client, err := cloudevents.NewDefaultClient()
	if err != nil {
		log.Fatal(err.Error())
	}
	// Initializing worker pool
	jobChannel := make(chan *work)
	for w := 1; w <= 4; w++ {
		go worker(w, jobChannel)
	}

	r := Receiver{client: client, jobChannel: jobChannel}
	if err := envconfig.Process("", &r); err != nil {
		log.Fatal(err.Error())
	}

	if err := client.StartReceiver(context.Background(), r.ReceiveAndReply); err != nil {
		log.Fatal(err)
	}
}

type Message struct {
	Payload []byte `json:"payload"`
}

type Request struct {
	Id         string `json:"id"`
	Embeddings string `json:"embeddings,omitempty"`
}

// ReceiveAndReply is invoked whenever we receive an event.
func (recv *Receiver) ReceiveAndReply(ctx context.Context, event cloudevents.Event) (*cloudevents.Event, cloudevents.Result) {
	req := Message{}
	if err := event.DataAs(&req); err != nil {
		log.Println(err)
		return nil, cloudevents.NewHTTPResult(400, "failed to convert data: %s", err)
	}

	enrollRequest := &face_authenticator.EnrollRequest{}

	if err := proto.Unmarshal(req.Payload, enrollRequest); err != nil {
		log.Println(err)
		return nil, cloudevents.NewHTTPResult(500, "failed to deserialize protobuf")
	}

	responseChannel := make(chan FeatureMatrix)
	recv.jobChannel <- &work{
		faceRequest:     enrollRequest.FaceRequest,
		responseChannel: responseChannel,
	}
	embeddings := <-responseChannel
	var serialized [authenticator.EMBEDDINGS_SIZE]float32
	ptr := &serialized[0]
	authenticator.Serialize_embeddings(embeddings, ptr)

	if err := recv.insertEmbeddings(ctx, enrollRequest.FaceRequest.Id, serialized); err != nil {
		log.Println(err)
		return nil, cloudevents.NewHTTPResult(500, "error while inserting to db")
	}

	enrollResponse := &face_authenticator.EnrollResponse{
		Status:  face_authenticator.EnrollStatus_ENROLL_STATUS_OK,
		Message: fmt.Sprintf("%s enrolled with success", enrollRequest.FaceRequest.Id),
	}
	resp, err := proto.Marshal(enrollResponse)
	if err != nil {
		log.Println(err)
		return nil, cloudevents.NewHTTPResult(500, "failed to serialize response")
	}

	msg := Message{Payload: resp}
	r := cloudevents.NewEvent(cloudevents.VersionV1)
	r.SetType("enroll-response")
	r.SetSource("enroller")
	if err := r.SetData("application/json", msg); err != nil {
		return nil, cloudevents.NewHTTPResult(500, "failed to set response data")
	}

	return &r, nil
}

func floatArrayToBytes(array [authenticator.EMBEDDINGS_SIZE]float32) []byte {
	var buf bytes.Buffer
	err := binary.Write(&buf, binary.LittleEndian, array)
	if err != nil {
		log.Println("binary.Write failed:", err)
	}
	return buf.Bytes()
}

func (recv *Receiver) insertEmbeddings(ctx context.Context, id string, embeddings [authenticator.EMBEDDINGS_SIZE]float32) error {
	r := cloudevents.NewEvent(cloudevents.VersionV1)
	r.SetType("insert")
	r.SetSource("enroller")

	payload := base64.StdEncoding.EncodeToString(floatArrayToBytes(embeddings))

	req := &Request{
		Id:         id,
		Embeddings: payload,
	}
	if err := r.SetData("application/json", req); err != nil {
		log.Println(err)
		return err
	}
	newCtx := cloudevents.ContextWithTarget(ctx, recv.Target)

	response, res := recv.client.Request(newCtx, r)
	if cloudevents.IsUndelivered(res) {
		log.Printf("Failed to request: %v", res)
		return res
	} else if response != nil {
		log.Printf("Got Event Response Context: %+v\n", response.Context)
	} else {
		log.Printf("Event sent at %s", time.Now())
		return errors.New("error insert embeddings failed")
	}
	return nil
}

type work struct {
	faceRequest     *face_authenticator.FaceRequest
	responseChannel chan FeatureMatrix
}

func validRectangle(coordinates *face_authenticator.FaceCoordinates) bool {
	return coordinates.TopLeft != nil && coordinates.TopLeft.X != 0 && coordinates.TopLeft.Y != 0 && coordinates.BottomRight != nil && coordinates.BottomRight.X != 0 && coordinates.BottomRight.Y != 0
}

func worker(idThread int, jobs <-chan *work) {
	authent := authenticator.NewAuthenticator(32)
	defer authenticator.DeleteAuthenticator(authent)
	log.Printf("Thread %d: Init authenticator\n", idThread)
	modelDir, pres := os.LookupEnv("model_dir")
	if !pres {
		modelDir = "/opt/enroller"
	}
	authent.Init(filepath.Join(modelDir, "shape_predictor_5_face_landmarks.dat"), filepath.Join(modelDir, "dlib_face_recognition_resnet_model_v1.dat"))
	log.Printf("Thread %d: Ready to authenticate\n", idThread)
	for job := range jobs {
		generateEmbeddings(&authent, job, idThread)
	}
}

func generateEmbeddings(authent *authenticator.Authenticator, work *work, idThread int) {
	facereq := work.faceRequest
	cImgData := authenticator.Load_mem_jpeg(&facereq.Face[0], len(facereq.Face))
	defer authenticator.DeleteImage(cImgData)
	var facePosition authenticator.Rectangle
	log.Printf("Thread %d: Searching for a face...\n", idThread)
	if coords := facereq.FaceCoordinates; coords == nil || !validRectangle(coords) {
		facePosition = (*authent).DetectFace(cImgData)
		defer authenticator.DeleteRectangle(facePosition)
	} else {
		facePosition = authenticator.NewRectangle()
		facePosition.SetTop(coords.TopLeft.Y)
		facePosition.SetLeft(coords.TopLeft.X)
		facePosition.SetBottom(coords.BottomRight.Y)
		facePosition.SetRight(coords.BottomRight.X)
	}
	log.Printf("Thread %d: Found face in area top_left(%d, %d), bottom_right(%d, %d)\n", idThread, facePosition.GetTop(), facePosition.GetLeft(), facePosition.GetBottom(), facePosition.GetRight())
	extractedFace := (*authent).ExtractFace(cImgData, facePosition)
	defer authenticator.DeleteImage(extractedFace)
	log.Printf("Thread %d: Generating embeddings\n", idThread)
	embeddings := (*authent).GenerateEmbeddings(extractedFace)
	work.responseChannel <- embeddings
}
