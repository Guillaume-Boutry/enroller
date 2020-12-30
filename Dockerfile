FROM registry.zouzland.com/face-authenticator-builder AS builder
WORKDIR /app
COPY go.* ./
RUN go mod download
COPY . .
RUN GOOS=linux go build -v ./cmd/enroller

FROM registry.zouzland.com/face-authenticator-runner
COPY models.txt models.txt
RUN wget -i models.txt --directory-prefix=/opt/enroller && bzip2 -d $(ls /opt/enroller/*.bz2)
COPY --from=builder /app/enroller /opt/enroller/enroller
# Run the web service on container startup.
CMD ["/opt/enroller/enroller"]